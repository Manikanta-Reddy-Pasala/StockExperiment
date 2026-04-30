# Latent View Analytics Ltd. (LATENTVIEW.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 291.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 5 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 5 / 3
- **Target hits / EMA400 exits:** 5 / 3
- **Total realized P&L (per unit):** 192.01
- **Avg P&L per closed trade:** 24.00

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 14:15:00 | 483.80 | 501.32 | 501.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 09:15:00 | 482.10 | 500.95 | 501.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 14:15:00 | 494.00 | 490.71 | 495.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-17 10:15:00 | 479.10 | 489.61 | 494.16 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 493.10 | 489.50 | 494.02 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-18 11:15:00 | 483.40 | 489.37 | 493.86 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 490.55 | 487.14 | 492.07 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-24 12:15:00 | 486.50 | 487.14 | 492.04 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 481.40 | 474.53 | 482.24 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-10-17 10:15:00 | 485.95 | 474.65 | 482.26 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 09:15:00 | 496.90 | 471.57 | 471.52 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 09:15:00 | 446.20 | 475.20 | 475.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 09:15:00 | 440.90 | 466.41 | 470.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 15:15:00 | 453.00 | 450.19 | 459.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-07 14:15:00 | 447.85 | 451.69 | 459.63 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 12:15:00 | 393.10 | 376.26 | 393.59 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-17 14:15:00 | 394.85 | 376.61 | 393.60 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 12:15:00 | 422.35 | 400.88 | 400.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 14:15:00 | 423.50 | 401.31 | 401.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 10:15:00 | 409.75 | 410.08 | 406.46 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-06 12:15:00 | 412.00 | 410.13 | 406.76 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-12 13:15:00 | 404.80 | 410.97 | 407.67 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 10:15:00 | 389.80 | 415.92 | 415.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 11:15:00 | 388.75 | 415.65 | 415.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 428.60 | 410.60 | 412.94 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 13:15:00 | 425.30 | 414.96 | 414.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-28 09:15:00 | 436.40 | 415.38 | 415.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-29 15:15:00 | 412.85 | 416.03 | 415.51 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-03 12:15:00 | 423.90 | 415.95 | 415.51 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-12 13:15:00 | 417.10 | 419.69 | 417.76 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 406.30 | 416.73 | 416.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 15:15:00 | 404.80 | 416.61 | 416.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 14:15:00 | 415.05 | 414.72 | 415.65 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 09:15:00 | 435.00 | 416.62 | 416.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 441.10 | 418.08 | 417.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 09:15:00 | 418.65 | 424.64 | 421.32 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-31 10:15:00 | 437.00 | 423.66 | 421.24 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 457.50 | 468.56 | 453.61 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-05 14:15:00 | 451.00 | 467.90 | 453.64 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 432.00 | 458.90 | 458.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 428.65 | 458.60 | 458.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 444.85 | 431.21 | 442.66 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-10 09:15:00 | 414.40 | 433.29 | 441.80 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-24 12:15:00 | 486.50 | 2024-10-03 09:15:00 | 469.87 | TARGET | 16.63 |
| SELL | 2024-09-18 11:15:00 | 483.40 | 2024-10-07 10:15:00 | 452.01 | TARGET | 31.39 |
| SELL | 2024-09-17 10:15:00 | 479.10 | 2024-10-17 10:15:00 | 485.95 | EXIT_EMA400 | -6.85 |
| SELL | 2025-02-07 14:15:00 | 447.85 | 2025-02-11 12:15:00 | 412.50 | TARGET | 35.35 |
| BUY | 2025-06-06 12:15:00 | 412.00 | 2025-06-12 13:15:00 | 404.80 | EXIT_EMA400 | -7.20 |
| BUY | 2025-09-03 12:15:00 | 423.90 | 2025-09-12 13:15:00 | 417.10 | EXIT_EMA400 | -6.80 |
| BUY | 2025-10-31 10:15:00 | 437.00 | 2025-11-11 11:15:00 | 484.29 | TARGET | 47.29 |
| SELL | 2026-02-10 09:15:00 | 414.40 | 2026-03-02 09:15:00 | 332.20 | TARGET | 82.20 |
