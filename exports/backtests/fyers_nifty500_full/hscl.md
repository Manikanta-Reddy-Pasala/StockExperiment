# Himadri Speciality Chemical Ltd. (HSCL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 608.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 3 |
| ENTRY1 | 7 |
| ENTRY2 | 1 |
| EXIT | 7 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 1 / 7
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -109.11
- **Avg P&L per closed trade:** -13.64

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 12:15:00 | 485.25 | 555.04 | 555.16 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 15:15:00 | 570.00 | 551.02 | 550.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 583.50 | 551.34 | 551.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 15:15:00 | 561.95 | 562.97 | 557.69 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-07 09:15:00 | 567.15 | 563.01 | 557.74 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 561.10 | 564.76 | 559.18 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-13 09:15:00 | 557.40 | 564.96 | 559.47 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 12:15:00 | 527.05 | 555.21 | 555.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 09:15:00 | 524.00 | 552.78 | 554.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 14:15:00 | 443.60 | 437.23 | 465.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-25 15:15:00 | 431.50 | 438.25 | 464.32 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-02 12:15:00 | 460.40 | 436.38 | 459.43 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 13:15:00 | 476.50 | 454.67 | 454.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 09:15:00 | 481.20 | 458.42 | 456.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 470.45 | 471.19 | 464.32 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-27 10:15:00 | 501.55 | 461.26 | 460.67 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 501.55 | 461.26 | 460.67 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-27 11:15:00 | 507.60 | 461.72 | 460.90 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-28 13:15:00 | 482.60 | 497.15 | 485.20 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 472.10 | 478.29 | 478.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 15:15:00 | 467.95 | 477.46 | 477.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 483.50 | 466.84 | 471.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-23 14:15:00 | 451.50 | 466.46 | 469.98 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-24 11:15:00 | 471.35 | 466.20 | 469.78 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 12:15:00 | 483.95 | 467.42 | 467.35 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 452.60 | 467.30 | 467.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 10:15:00 | 449.80 | 461.61 | 464.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 454.50 | 451.26 | 457.28 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 15:15:00 | 484.00 | 460.94 | 460.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 486.35 | 461.20 | 461.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 13:15:00 | 473.10 | 474.03 | 468.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-08 14:15:00 | 476.00 | 474.05 | 468.82 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 474.35 | 474.06 | 468.88 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-12 09:15:00 | 464.35 | 473.86 | 468.95 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 11:15:00 | 450.70 | 466.05 | 466.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 12:15:00 | 449.70 | 465.89 | 466.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 15:15:00 | 470.50 | 464.52 | 465.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-01 12:15:00 | 453.15 | 463.99 | 465.01 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-02 12:15:00 | 469.50 | 463.19 | 464.56 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 486.65 | 464.37 | 464.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 12:15:00 | 488.55 | 464.83 | 464.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 469.65 | 470.29 | 467.56 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 435.50 | 464.96 | 465.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 426.50 | 457.79 | 461.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 13:15:00 | 457.20 | 455.47 | 459.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-18 15:15:00 | 452.50 | 455.43 | 459.61 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-03-25 09:15:00 | 461.15 | 452.62 | 457.55 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 494.15 | 459.05 | 459.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 505.90 | 463.20 | 461.19 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-01-07 09:15:00 | 567.15 | 2025-01-13 09:15:00 | 557.40 | EXIT_EMA400 | -9.75 |
| SELL | 2025-03-25 15:15:00 | 431.50 | 2025-04-02 12:15:00 | 460.40 | EXIT_EMA400 | -28.90 |
| BUY | 2025-06-27 10:15:00 | 501.55 | 2025-07-28 13:15:00 | 482.60 | EXIT_EMA400 | -18.95 |
| BUY | 2025-06-27 11:15:00 | 507.60 | 2025-07-28 13:15:00 | 482.60 | EXIT_EMA400 | -25.00 |
| SELL | 2025-09-23 14:15:00 | 451.50 | 2025-09-24 11:15:00 | 471.35 | EXIT_EMA400 | -19.85 |
| BUY | 2026-01-08 14:15:00 | 476.00 | 2026-01-12 09:15:00 | 464.35 | EXIT_EMA400 | -11.65 |
| SELL | 2026-02-01 12:15:00 | 453.15 | 2026-02-02 12:15:00 | 469.50 | EXIT_EMA400 | -16.35 |
| SELL | 2026-03-18 15:15:00 | 452.50 | 2026-03-23 12:15:00 | 431.16 | TARGET | 21.34 |
