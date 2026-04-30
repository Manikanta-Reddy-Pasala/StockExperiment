# Coal India Ltd. (COALINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 481.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 4 |
| ENTRY1 | 7 |
| ENTRY2 | 1 |
| EXIT | 7 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 1 / 7
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -14.24
- **Avg P&L per closed trade:** -1.78

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 15:15:00 | 487.90 | 500.44 | 500.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 485.00 | 497.35 | 498.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 10:15:00 | 387.00 | 385.32 | 401.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-31 09:15:00 | 379.65 | 385.26 | 401.37 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-06 12:15:00 | 383.05 | 369.62 | 381.65 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 398.00 | 386.80 | 386.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 398.80 | 386.92 | 386.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 385.55 | 387.51 | 387.12 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 12:15:00 | 369.80 | 386.69 | 386.73 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 11:15:00 | 398.35 | 386.76 | 386.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 13:15:00 | 399.60 | 387.01 | 386.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 11:15:00 | 391.25 | 391.58 | 389.50 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 13:15:00 | 394.15 | 388.72 | 388.38 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 394.50 | 397.08 | 393.91 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-03 10:15:00 | 393.30 | 397.04 | 393.90 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 11:15:00 | 387.05 | 393.34 | 393.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 10:15:00 | 386.00 | 392.94 | 393.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 14:15:00 | 388.80 | 388.77 | 390.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-21 09:15:00 | 386.90 | 388.75 | 390.56 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 389.85 | 388.60 | 390.39 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-07-23 09:15:00 | 390.80 | 388.64 | 390.38 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 15:15:00 | 394.95 | 385.98 | 385.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 396.40 | 386.08 | 385.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 389.40 | 389.80 | 388.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-29 10:15:00 | 391.35 | 389.81 | 388.23 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-29 12:15:00 | 386.30 | 389.78 | 388.22 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 382.50 | 387.25 | 387.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 381.40 | 387.14 | 387.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 12:15:00 | 387.40 | 386.75 | 386.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-17 13:15:00 | 385.90 | 386.78 | 386.99 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-17 14:15:00 | 388.60 | 386.80 | 387.00 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 392.15 | 387.20 | 387.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 11:15:00 | 393.70 | 387.32 | 387.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 13:15:00 | 382.60 | 389.11 | 388.23 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 13:15:00 | 374.35 | 387.52 | 387.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 372.95 | 387.38 | 387.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 12:15:00 | 386.80 | 385.73 | 386.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-13 12:15:00 | 384.95 | 385.76 | 386.55 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 384.95 | 385.76 | 386.55 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-13 13:15:00 | 383.85 | 385.74 | 386.53 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 385.55 | 385.67 | 386.48 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-14 14:15:00 | 386.95 | 385.67 | 386.47 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 12:15:00 | 402.70 | 384.05 | 384.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 405.35 | 384.80 | 384.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 419.30 | 419.57 | 408.02 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-02 14:15:00 | 423.85 | 419.44 | 408.30 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-02-13 09:15:00 | 412.45 | 422.91 | 413.16 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-31 09:15:00 | 379.65 | 2025-03-06 12:15:00 | 383.05 | EXIT_EMA400 | -3.40 |
| BUY | 2025-05-12 13:15:00 | 394.15 | 2025-05-20 09:15:00 | 411.46 | TARGET | 17.31 |
| SELL | 2025-07-21 09:15:00 | 386.90 | 2025-07-23 09:15:00 | 390.80 | EXIT_EMA400 | -3.90 |
| BUY | 2025-09-29 10:15:00 | 391.35 | 2025-09-29 12:15:00 | 386.30 | EXIT_EMA400 | -5.05 |
| SELL | 2025-10-17 13:15:00 | 385.90 | 2025-10-17 14:15:00 | 388.60 | EXIT_EMA400 | -2.70 |
| SELL | 2025-11-13 12:15:00 | 384.95 | 2025-11-14 14:15:00 | 386.95 | EXIT_EMA400 | -2.00 |
| SELL | 2025-11-13 13:15:00 | 383.85 | 2025-11-14 14:15:00 | 386.95 | EXIT_EMA400 | -3.10 |
| BUY | 2026-02-02 14:15:00 | 423.85 | 2026-02-13 09:15:00 | 412.45 | EXIT_EMA400 | -11.40 |
