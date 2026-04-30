# Himadri Speciality Chemical Ltd. (HSCL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 607.75
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 16 |
| ALERT1 | 12 |
| ALERT2 | 11 |
| ALERT3 | 5 |
| ENTRY1 | 8 |
| ENTRY2 | 2 |
| EXIT | 8 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 1 / 9
- **Target hits / EMA400 exits:** 1 / 9
- **Total realized P&L (per unit):** -144.70
- **Avg P&L per closed trade:** -14.47

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 12:15:00 | 316.00 | 334.63 | 334.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-22 14:15:00 | 314.00 | 334.23 | 334.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-03 11:15:00 | 330.00 | 327.32 | 330.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-10 14:15:00 | 321.10 | 328.62 | 330.75 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-19 09:15:00 | 335.85 | 325.67 | 328.86 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 13:15:00 | 367.80 | 331.63 | 331.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 09:15:00 | 376.00 | 334.99 | 333.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 11:15:00 | 346.75 | 347.25 | 340.53 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-07 13:15:00 | 363.00 | 347.40 | 340.67 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 13:15:00 | 344.15 | 348.42 | 341.66 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-05-09 14:15:00 | 349.55 | 348.43 | 341.70 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 10:15:00 | 342.10 | 348.37 | 341.77 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-05-10 12:15:00 | 340.95 | 348.25 | 341.78 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 13:15:00 | 487.65 | 554.52 | 554.82 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 11:15:00 | 555.80 | 551.11 | 551.11 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 13:15:00 | 545.85 | 551.08 | 551.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 537.85 | 550.85 | 550.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 15:15:00 | 552.60 | 549.69 | 550.35 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 15:15:00 | 570.05 | 551.08 | 551.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 583.50 | 551.41 | 551.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 15:15:00 | 562.30 | 563.05 | 557.73 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-07 09:15:00 | 567.50 | 563.09 | 557.78 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 561.50 | 564.83 | 559.22 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-13 09:15:00 | 557.40 | 565.01 | 559.50 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 12:15:00 | 527.50 | 555.26 | 555.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 09:15:00 | 524.00 | 552.82 | 554.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 14:15:00 | 443.40 | 437.50 | 466.40 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-25 15:15:00 | 431.10 | 438.48 | 464.83 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-02 12:15:00 | 460.40 | 436.55 | 459.87 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 14:15:00 | 477.00 | 454.87 | 454.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 09:15:00 | 481.20 | 458.39 | 456.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 470.30 | 471.18 | 464.38 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-27 10:15:00 | 501.55 | 461.28 | 460.72 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 501.55 | 461.28 | 460.72 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-27 11:15:00 | 507.75 | 461.74 | 460.96 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-28 13:15:00 | 482.60 | 497.16 | 485.23 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 472.10 | 478.29 | 478.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 11:15:00 | 469.00 | 478.13 | 478.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 14:15:00 | 479.65 | 477.72 | 478.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-22 10:15:00 | 471.95 | 477.80 | 478.03 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-10 09:15:00 | 483.50 | 466.83 | 471.42 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 12:15:00 | 483.95 | 467.43 | 467.36 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 452.60 | 467.30 | 467.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 10:15:00 | 449.80 | 461.60 | 464.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 454.50 | 451.26 | 457.28 | EMA200 retest candle locked |

### Cycle 12 — BUY (started 2025-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 15:15:00 | 484.00 | 460.96 | 460.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 486.00 | 461.21 | 461.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 13:15:00 | 473.10 | 474.05 | 468.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-08 14:15:00 | 476.00 | 474.07 | 468.84 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 474.35 | 474.10 | 468.90 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-12 09:15:00 | 464.50 | 473.89 | 468.98 | Close below EMA400 |

### Cycle 13 — SELL (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 11:15:00 | 450.70 | 466.08 | 466.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 12:15:00 | 449.70 | 465.92 | 466.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 15:15:00 | 470.25 | 464.55 | 465.35 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-02 09:15:00 | 447.40 | 464.12 | 465.10 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-02 12:15:00 | 469.95 | 463.92 | 464.98 | Close above EMA400 |

### Cycle 14 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 486.65 | 464.62 | 464.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 12:15:00 | 488.55 | 465.08 | 464.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 469.65 | 470.45 | 467.75 | EMA200 retest candle locked |

### Cycle 15 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 431.35 | 465.37 | 465.40 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 10:15:00 | 492.15 | 459.47 | 459.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 505.90 | 462.27 | 460.76 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-04-10 14:15:00 | 321.10 | 2024-04-19 09:15:00 | 335.85 | EXIT_EMA400 | -14.75 |
| BUY | 2024-05-07 13:15:00 | 363.00 | 2024-05-10 12:15:00 | 340.95 | EXIT_EMA400 | -22.05 |
| BUY | 2024-05-09 14:15:00 | 349.55 | 2024-05-10 12:15:00 | 340.95 | EXIT_EMA400 | -8.60 |
| BUY | 2025-01-07 09:15:00 | 567.50 | 2025-01-13 09:15:00 | 557.40 | EXIT_EMA400 | -10.10 |
| SELL | 2025-03-25 15:15:00 | 431.10 | 2025-04-02 12:15:00 | 460.40 | EXIT_EMA400 | -29.30 |
| BUY | 2025-06-27 10:15:00 | 501.55 | 2025-07-28 13:15:00 | 482.60 | EXIT_EMA400 | -18.95 |
| BUY | 2025-06-27 11:15:00 | 507.75 | 2025-07-28 13:15:00 | 482.60 | EXIT_EMA400 | -25.15 |
| SELL | 2025-08-22 10:15:00 | 471.95 | 2025-08-26 15:15:00 | 453.70 | TARGET | 18.25 |
| BUY | 2026-01-08 14:15:00 | 476.00 | 2026-01-12 09:15:00 | 464.50 | EXIT_EMA400 | -11.50 |
| SELL | 2026-02-02 09:15:00 | 447.40 | 2026-02-02 12:15:00 | 469.95 | EXIT_EMA400 | -22.55 |
