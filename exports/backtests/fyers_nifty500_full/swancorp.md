# Swan Corp Ltd. (SWANCORP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 335.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 6 |
| ENTRY1 | 8 |
| ENTRY2 | 4 |
| EXIT | 7 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 2
- **Winners / losers:** 0 / 10
- **Target hits / EMA400 exits:** 0 / 10
- **Total realized P&L (per unit):** -292.25
- **Avg P&L per closed trade:** -29.23

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 10:15:00 | 599.80 | 655.54 | 655.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 10:15:00 | 594.90 | 651.68 | 653.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 10:15:00 | 543.20 | 529.27 | 565.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 15:15:00 | 522.20 | 530.20 | 563.90 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 548.30 | 529.68 | 558.07 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-19 14:15:00 | 538.80 | 530.56 | 557.82 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-11-22 13:15:00 | 569.25 | 530.87 | 556.26 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 09:15:00 | 723.90 | 575.52 | 574.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 10:15:00 | 752.90 | 611.53 | 594.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 13:15:00 | 703.90 | 704.60 | 663.77 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-07 09:15:00 | 717.50 | 704.58 | 664.36 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-10 09:15:00 | 663.50 | 703.42 | 667.82 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 15:15:00 | 529.30 | 648.98 | 649.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 517.55 | 600.45 | 621.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 09:15:00 | 453.15 | 451.22 | 501.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-26 12:15:00 | 438.55 | 451.45 | 497.77 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 438.30 | 422.87 | 446.30 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-05-15 13:15:00 | 434.55 | 423.28 | 446.16 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 441.50 | 423.94 | 446.04 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-05-16 13:15:00 | 456.10 | 424.67 | 446.07 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 11:15:00 | 455.45 | 443.97 | 443.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 464.15 | 444.46 | 444.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 12:15:00 | 472.40 | 472.80 | 460.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-29 12:15:00 | 477.90 | 472.68 | 461.27 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 471.40 | 473.40 | 462.25 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-01 09:15:00 | 459.40 | 473.07 | 462.47 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 13:15:00 | 429.70 | 454.33 | 454.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 14:15:00 | 426.60 | 454.06 | 454.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 13:15:00 | 463.05 | 448.30 | 451.10 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 11:15:00 | 469.20 | 453.67 | 453.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 09:15:00 | 475.00 | 454.95 | 454.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 09:15:00 | 463.60 | 467.14 | 461.63 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 433.50 | 457.98 | 458.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 432.00 | 457.03 | 457.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 14:15:00 | 449.85 | 446.43 | 451.39 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-06 09:15:00 | 441.80 | 454.86 | 455.14 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 441.80 | 454.86 | 455.14 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-06 11:15:00 | 438.55 | 454.57 | 454.98 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-11-07 10:15:00 | 473.50 | 453.92 | 454.64 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 14:15:00 | 474.95 | 455.42 | 455.37 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 446.30 | 456.33 | 456.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 442.05 | 455.66 | 456.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 14:15:00 | 454.40 | 454.14 | 455.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-08 09:15:00 | 436.95 | 453.50 | 454.79 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-09 14:15:00 | 458.00 | 452.37 | 454.13 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 13:15:00 | 471.30 | 455.49 | 455.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 476.00 | 456.78 | 456.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 13:15:00 | 461.50 | 462.27 | 459.40 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-05 09:15:00 | 474.70 | 462.39 | 459.50 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-08 10:15:00 | 459.25 | 463.77 | 460.54 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 11:15:00 | 420.10 | 457.62 | 457.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 12:15:00 | 419.25 | 457.24 | 457.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 10:15:00 | 364.00 | 338.72 | 364.61 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-24 10:15:00 | 329.35 | 342.16 | 361.67 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 356.35 | 341.26 | 359.33 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-30 09:15:00 | 337.00 | 341.55 | 358.85 | Sell entry 2 (retest2 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-08 15:15:00 | 522.20 | 2024-11-22 13:15:00 | 569.25 | EXIT_EMA400 | -47.05 |
| SELL | 2024-11-19 14:15:00 | 538.80 | 2024-11-22 13:15:00 | 569.25 | EXIT_EMA400 | -30.45 |
| BUY | 2025-01-07 09:15:00 | 717.50 | 2025-01-10 09:15:00 | 663.50 | EXIT_EMA400 | -54.00 |
| SELL | 2025-03-26 12:15:00 | 438.55 | 2025-05-16 13:15:00 | 456.10 | EXIT_EMA400 | -17.55 |
| SELL | 2025-05-15 13:15:00 | 434.55 | 2025-05-16 13:15:00 | 456.10 | EXIT_EMA400 | -21.55 |
| BUY | 2025-07-29 12:15:00 | 477.90 | 2025-08-01 09:15:00 | 459.40 | EXIT_EMA400 | -18.50 |
| SELL | 2025-11-06 09:15:00 | 441.80 | 2025-11-07 10:15:00 | 473.50 | EXIT_EMA400 | -31.70 |
| SELL | 2025-11-06 11:15:00 | 438.55 | 2025-11-07 10:15:00 | 473.50 | EXIT_EMA400 | -34.95 |
| SELL | 2025-12-08 09:15:00 | 436.95 | 2025-12-09 14:15:00 | 458.00 | EXIT_EMA400 | -21.05 |
| BUY | 2026-01-05 09:15:00 | 474.70 | 2026-01-08 10:15:00 | 459.25 | EXIT_EMA400 | -15.45 |
