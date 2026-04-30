# Transformers And Rectifiers (India) Ltd. (TARIL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 333.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT3 | 7 |
| ENTRY1 | 7 |
| ENTRY2 | 7 |
| EXIT | 7 |

## P&L

- **Trades closed:** 14
- **Trades open at end:** 0
- **Winners / losers:** 4 / 10
- **Target hits / EMA400 exits:** 4 / 10
- **Total realized P&L (per unit):** 43.21
- **Avg P&L per closed trade:** 3.09

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 12:15:00 | 342.38 | 354.40 | 354.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 09:15:00 | 332.48 | 353.80 | 354.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-03 09:15:00 | 341.68 | 334.02 | 342.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-08 09:15:00 | 317.65 | 333.10 | 340.76 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-10-09 09:15:00 | 353.90 | 333.16 | 340.52 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 10:15:00 | 430.13 | 346.69 | 346.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 09:15:00 | 447.00 | 351.71 | 349.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-14 10:15:00 | 425.60 | 432.67 | 402.39 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-11-18 10:15:00 | 451.95 | 432.97 | 403.58 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-14 09:15:00 | 507.25 | 554.79 | 514.15 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 13:15:00 | 401.00 | 495.68 | 496.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 09:15:00 | 392.25 | 492.80 | 494.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 11:15:00 | 450.90 | 439.99 | 461.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-25 11:15:00 | 433.50 | 440.21 | 460.66 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 430.75 | 439.66 | 459.88 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-27 11:15:00 | 420.00 | 439.32 | 459.50 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-19 10:15:00 | 436.75 | 413.81 | 436.46 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 14:15:00 | 509.40 | 452.57 | 452.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 10:15:00 | 511.95 | 454.21 | 453.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-28 14:15:00 | 504.60 | 505.76 | 485.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-29 09:15:00 | 512.10 | 505.85 | 485.84 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 14:15:00 | 495.70 | 505.15 | 486.65 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-02 09:15:00 | 499.80 | 504.96 | 486.74 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 488.75 | 504.33 | 487.05 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-05 12:15:00 | 501.95 | 504.21 | 487.24 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-05-06 14:15:00 | 483.20 | 503.47 | 487.61 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 471.80 | 498.28 | 498.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 467.50 | 497.97 | 498.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 492.50 | 491.73 | 494.76 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 11:15:00 | 514.00 | 497.43 | 497.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 518.85 | 498.28 | 497.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 09:15:00 | 500.90 | 501.80 | 499.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-29 13:15:00 | 510.40 | 499.02 | 498.58 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 510.40 | 499.02 | 498.58 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-31 12:15:00 | 513.35 | 500.29 | 499.26 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 505.45 | 507.20 | 503.03 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-06 11:15:00 | 512.70 | 507.25 | 503.07 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-07 09:15:00 | 499.10 | 507.24 | 503.17 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 15:15:00 | 496.90 | 502.45 | 502.46 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 505.85 | 502.48 | 502.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 509.10 | 502.55 | 502.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 11:15:00 | 501.75 | 503.30 | 502.90 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-15 10:15:00 | 506.00 | 503.05 | 502.79 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 506.00 | 503.05 | 502.79 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-16 09:15:00 | 517.60 | 503.29 | 502.92 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-09-25 14:15:00 | 506.65 | 513.15 | 508.63 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 490.40 | 505.05 | 505.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 487.35 | 504.27 | 504.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 09:15:00 | 304.15 | 302.72 | 360.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-12 09:15:00 | 266.60 | 301.84 | 331.40 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 287.86 | 261.22 | 289.69 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-10 09:15:00 | 277.60 | 261.57 | 289.59 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-11 09:15:00 | 294.20 | 263.00 | 289.34 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 15:15:00 | 322.40 | 288.75 | 288.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 326.55 | 289.12 | 288.88 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-08 09:15:00 | 317.65 | 2024-10-09 09:15:00 | 353.90 | EXIT_EMA400 | -36.25 |
| BUY | 2024-11-18 10:15:00 | 451.95 | 2024-12-17 09:15:00 | 597.06 | TARGET | 145.11 |
| SELL | 2025-02-25 11:15:00 | 433.50 | 2025-03-19 10:15:00 | 436.75 | EXIT_EMA400 | -3.25 |
| SELL | 2025-02-27 11:15:00 | 420.00 | 2025-03-19 10:15:00 | 436.75 | EXIT_EMA400 | -16.75 |
| BUY | 2025-04-29 09:15:00 | 512.10 | 2025-05-06 14:15:00 | 483.20 | EXIT_EMA400 | -28.90 |
| BUY | 2025-05-02 09:15:00 | 499.80 | 2025-05-06 14:15:00 | 483.20 | EXIT_EMA400 | -16.60 |
| BUY | 2025-05-05 12:15:00 | 501.95 | 2025-05-06 14:15:00 | 483.20 | EXIT_EMA400 | -18.75 |
| BUY | 2025-07-29 13:15:00 | 510.40 | 2025-08-04 12:15:00 | 545.87 | TARGET | 35.47 |
| BUY | 2025-07-31 12:15:00 | 513.35 | 2025-08-04 13:15:00 | 555.62 | TARGET | 42.27 |
| BUY | 2025-08-06 11:15:00 | 512.70 | 2025-08-07 09:15:00 | 499.10 | EXIT_EMA400 | -13.60 |
| BUY | 2025-09-15 10:15:00 | 506.00 | 2025-09-16 09:15:00 | 515.62 | TARGET | 9.62 |
| BUY | 2025-09-16 09:15:00 | 517.60 | 2025-09-25 14:15:00 | 506.65 | EXIT_EMA400 | -10.95 |
| SELL | 2026-01-12 09:15:00 | 266.60 | 2026-02-11 09:15:00 | 294.20 | EXIT_EMA400 | -27.60 |
| SELL | 2026-02-10 09:15:00 | 277.60 | 2026-02-11 09:15:00 | 294.20 | EXIT_EMA400 | -16.60 |
