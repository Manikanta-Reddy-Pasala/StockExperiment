# Coal India Ltd. (COALINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 481.45
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 3 |
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| EXIT | 8 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** -2.01
- **Avg P&L per closed trade:** -0.25

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 11:15:00 | 249.85 | 232.89 | 232.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 13:15:00 | 251.20 | 233.23 | 232.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 12:15:00 | 430.70 | 431.04 | 404.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-28 13:15:00 | 434.90 | 431.08 | 404.52 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-13 14:15:00 | 415.45 | 439.52 | 416.70 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 09:15:00 | 489.95 | 500.35 | 500.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 11:15:00 | 488.45 | 500.14 | 500.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 09:15:00 | 501.55 | 499.20 | 499.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-15 13:15:00 | 490.80 | 499.01 | 499.64 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-06 12:15:00 | 383.00 | 369.57 | 381.85 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 10:15:00 | 397.70 | 387.01 | 386.98 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 11:15:00 | 369.75 | 386.86 | 386.92 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 12:15:00 | 398.80 | 386.88 | 386.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 13:15:00 | 399.75 | 387.01 | 386.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 11:15:00 | 391.45 | 391.58 | 389.56 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 13:15:00 | 394.15 | 388.72 | 388.43 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 394.50 | 397.10 | 393.95 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-03 10:15:00 | 393.30 | 397.06 | 393.94 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 11:15:00 | 387.05 | 393.35 | 393.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 10:15:00 | 386.00 | 392.95 | 393.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 14:15:00 | 388.80 | 388.77 | 390.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-21 09:15:00 | 386.90 | 388.75 | 390.56 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 389.85 | 388.59 | 390.40 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-07-23 09:15:00 | 390.75 | 388.64 | 390.38 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 15:15:00 | 394.75 | 385.98 | 385.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 396.50 | 386.08 | 385.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 389.50 | 389.80 | 388.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-29 10:15:00 | 391.35 | 389.81 | 388.23 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-29 12:15:00 | 386.35 | 389.77 | 388.22 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 382.70 | 387.25 | 387.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 382.05 | 387.19 | 387.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 12:15:00 | 387.40 | 386.75 | 386.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-17 13:15:00 | 385.90 | 386.78 | 386.99 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-17 14:15:00 | 388.60 | 386.80 | 387.00 | Close above EMA400 |

### Cycle 9 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 392.15 | 387.20 | 387.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 11:15:00 | 393.65 | 387.31 | 387.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 13:15:00 | 382.60 | 389.11 | 388.23 | EMA200 retest candle locked |

### Cycle 10 — SELL (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 13:15:00 | 374.35 | 387.52 | 387.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 372.90 | 387.38 | 387.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 12:15:00 | 386.80 | 385.72 | 386.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-13 13:15:00 | 383.85 | 385.73 | 386.53 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 385.55 | 385.67 | 386.48 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-14 14:15:00 | 386.95 | 385.67 | 386.46 | Close above EMA400 |

### Cycle 11 — BUY (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 12:15:00 | 402.70 | 384.05 | 384.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 405.35 | 384.79 | 384.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 416.35 | 418.98 | 407.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-02 14:15:00 | 423.85 | 418.91 | 407.63 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-02-13 09:15:00 | 412.25 | 422.60 | 412.65 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-02-28 13:15:00 | 434.90 | 2024-03-13 14:15:00 | 415.45 | EXIT_EMA400 | -19.45 |
| SELL | 2024-10-15 13:15:00 | 490.80 | 2024-10-23 09:15:00 | 464.27 | TARGET | 26.53 |
| BUY | 2025-05-12 13:15:00 | 394.15 | 2025-05-20 09:15:00 | 411.30 | TARGET | 17.15 |
| SELL | 2025-07-21 09:15:00 | 386.90 | 2025-07-23 09:15:00 | 390.75 | EXIT_EMA400 | -3.85 |
| BUY | 2025-09-29 10:15:00 | 391.35 | 2025-09-29 12:15:00 | 386.35 | EXIT_EMA400 | -5.00 |
| SELL | 2025-10-17 13:15:00 | 385.90 | 2025-10-17 14:15:00 | 388.60 | EXIT_EMA400 | -2.70 |
| SELL | 2025-11-13 13:15:00 | 383.85 | 2025-11-14 14:15:00 | 386.95 | EXIT_EMA400 | -3.10 |
| BUY | 2026-02-02 14:15:00 | 423.85 | 2026-02-13 09:15:00 | 412.25 | EXIT_EMA400 | -11.60 |
