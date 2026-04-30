# PCBL Chemical Ltd. (PCBL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 290.58
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 5 |
| ENTRY1 | 7 |
| ENTRY2 | 2 |
| EXIT | 7 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 1 / 8
- **Target hits / EMA400 exits:** 1 / 8
- **Total realized P&L (per unit):** -125.06
- **Avg P&L per closed trade:** -13.90

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 12:15:00 | 247.75 | 277.73 | 277.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 15:15:00 | 246.80 | 276.84 | 277.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 09:15:00 | 271.65 | 271.10 | 274.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-28 11:15:00 | 269.35 | 271.09 | 273.99 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-01 11:15:00 | 274.25 | 271.05 | 273.87 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 10:15:00 | 282.05 | 259.20 | 259.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 10:15:00 | 284.05 | 260.49 | 259.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 14:15:00 | 503.60 | 504.91 | 451.22 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-09 09:15:00 | 517.55 | 504.91 | 453.58 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 465.75 | 500.57 | 464.91 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-22 09:15:00 | 447.65 | 499.70 | 464.82 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 11:15:00 | 396.40 | 447.83 | 447.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-14 14:15:00 | 391.70 | 446.26 | 447.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 12:15:00 | 433.90 | 427.75 | 436.44 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 09:15:00 | 486.30 | 441.42 | 441.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 10:15:00 | 490.20 | 441.91 | 441.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-23 12:15:00 | 457.50 | 457.57 | 450.73 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-26 10:15:00 | 462.95 | 457.69 | 451.18 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-12-30 12:15:00 | 449.45 | 457.88 | 451.79 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 12:15:00 | 390.20 | 448.18 | 448.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 386.65 | 447.56 | 447.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 13:15:00 | 397.00 | 392.31 | 412.41 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-11 09:15:00 | 380.55 | 397.28 | 411.69 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 390.65 | 382.21 | 395.02 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-10 10:15:00 | 386.00 | 382.24 | 394.98 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 10:15:00 | 389.90 | 381.04 | 391.90 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-20 12:15:00 | 392.00 | 381.82 | 391.83 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 13:15:00 | 432.50 | 399.34 | 399.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 434.90 | 399.70 | 399.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 381.20 | 403.31 | 401.38 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-08 09:15:00 | 398.85 | 402.55 | 401.06 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 398.85 | 402.55 | 401.06 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-04-08 14:15:00 | 399.55 | 402.56 | 401.10 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 15:15:00 | 356.70 | 402.03 | 402.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 354.00 | 390.96 | 396.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 14:15:00 | 391.95 | 386.36 | 392.77 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 13:15:00 | 417.50 | 396.00 | 395.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 429.50 | 396.77 | 396.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 401.30 | 402.31 | 399.54 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-25 10:15:00 | 408.40 | 398.27 | 398.00 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 408.40 | 398.27 | 398.00 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-25 14:15:00 | 410.00 | 398.61 | 398.18 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-07 15:15:00 | 401.80 | 404.69 | 401.84 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 388.00 | 404.09 | 404.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 12:15:00 | 385.40 | 403.90 | 404.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 10:15:00 | 387.60 | 385.89 | 391.83 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-09 15:15:00 | 384.85 | 385.86 | 391.66 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-10 09:15:00 | 396.00 | 385.96 | 391.69 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-03-28 11:15:00 | 269.35 | 2024-04-01 11:15:00 | 274.25 | EXIT_EMA400 | -4.90 |
| BUY | 2024-10-09 09:15:00 | 517.55 | 2024-10-22 09:15:00 | 447.65 | EXIT_EMA400 | -69.90 |
| BUY | 2024-12-26 10:15:00 | 462.95 | 2024-12-30 12:15:00 | 449.45 | EXIT_EMA400 | -13.50 |
| SELL | 2025-02-11 09:15:00 | 380.55 | 2025-03-20 12:15:00 | 392.00 | EXIT_EMA400 | -11.45 |
| SELL | 2025-03-10 10:15:00 | 386.00 | 2025-03-20 12:15:00 | 392.00 | EXIT_EMA400 | -6.00 |
| BUY | 2025-04-08 09:15:00 | 398.85 | 2025-04-08 12:15:00 | 405.49 | TARGET | 6.64 |
| BUY | 2025-06-25 10:15:00 | 408.40 | 2025-07-07 15:15:00 | 401.80 | EXIT_EMA400 | -6.60 |
| BUY | 2025-06-25 14:15:00 | 410.00 | 2025-07-07 15:15:00 | 401.80 | EXIT_EMA400 | -8.20 |
| SELL | 2025-09-09 15:15:00 | 384.85 | 2025-09-10 09:15:00 | 396.00 | EXIT_EMA400 | -11.15 |
