# General Insurance Corporation of India (GICRE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 392.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 16 |
| ALERT1 | 12 |
| ALERT2 | 11 |
| ALERT3 | 6 |
| ENTRY1 | 7 |
| ENTRY2 | 1 |
| EXIT | 7 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 1 / 7
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -71.87
- **Avg P&L per closed trade:** -8.98

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 09:15:00 | 385.30 | 391.83 | 391.84 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 14:15:00 | 399.75 | 391.90 | 391.86 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 14:15:00 | 383.55 | 391.86 | 391.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 15:15:00 | 380.30 | 391.75 | 391.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 13:15:00 | 379.85 | 377.49 | 383.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-07 13:15:00 | 375.90 | 377.63 | 383.08 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-22 09:15:00 | 378.75 | 371.62 | 378.26 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 09:15:00 | 413.30 | 383.04 | 383.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 09:15:00 | 414.75 | 386.85 | 384.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 14:15:00 | 434.50 | 436.33 | 418.33 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-07 09:15:00 | 449.45 | 436.47 | 418.58 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 425.60 | 439.19 | 422.40 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-13 10:15:00 | 419.85 | 438.99 | 422.38 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 389.10 | 417.47 | 417.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 10:15:00 | 387.00 | 417.17 | 417.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 13:15:00 | 389.00 | 387.78 | 397.01 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 09:15:00 | 417.05 | 403.38 | 403.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 423.60 | 405.05 | 404.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 14:15:00 | 405.95 | 406.43 | 404.96 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 12:15:00 | 392.50 | 403.55 | 403.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 13:15:00 | 391.15 | 403.43 | 403.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 09:15:00 | 404.00 | 402.94 | 403.27 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 10:15:00 | 417.80 | 403.61 | 403.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 09:15:00 | 429.35 | 404.53 | 404.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 15:15:00 | 415.05 | 415.29 | 410.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-02 09:15:00 | 424.80 | 415.39 | 410.44 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-06 14:15:00 | 409.35 | 416.76 | 411.62 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 401.65 | 412.60 | 412.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 398.60 | 412.47 | 412.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 386.25 | 385.67 | 394.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-18 10:15:00 | 382.20 | 386.19 | 393.71 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 386.40 | 382.55 | 389.77 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-07-31 14:15:00 | 391.95 | 382.78 | 389.60 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 14:15:00 | 383.50 | 378.23 | 378.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 10:15:00 | 385.35 | 378.41 | 378.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 377.85 | 379.04 | 378.64 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 371.70 | 378.31 | 378.32 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 11:15:00 | 381.40 | 378.32 | 378.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 12:15:00 | 382.05 | 378.36 | 378.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 382.00 | 383.08 | 381.02 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-26 09:15:00 | 390.00 | 382.68 | 381.01 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 390.00 | 382.68 | 381.01 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-11-26 11:15:00 | 391.10 | 382.84 | 381.11 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 384.20 | 384.30 | 382.31 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-05 11:15:00 | 381.75 | 384.27 | 382.31 | Close below EMA400 |

### Cycle 13 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 364.05 | 380.87 | 380.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 14:15:00 | 362.40 | 380.02 | 380.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 377.35 | 377.35 | 378.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-26 12:15:00 | 374.00 | 377.31 | 378.92 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 378.05 | 376.41 | 378.32 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-30 15:15:00 | 386.25 | 376.51 | 378.36 | Close above EMA400 |

### Cycle 14 — BUY (started 2026-02-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 13:15:00 | 393.55 | 376.12 | 376.06 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2026-03-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 13:15:00 | 364.60 | 377.68 | 377.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-10 11:15:00 | 363.50 | 377.07 | 377.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 10:15:00 | 375.10 | 373.37 | 375.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-19 09:15:00 | 368.25 | 373.32 | 375.23 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 371.35 | 368.92 | 372.36 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-01 10:15:00 | 377.00 | 369.00 | 372.39 | Close above EMA400 |

### Cycle 16 — BUY (started 2026-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 13:15:00 | 397.50 | 375.01 | 374.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 417.00 | 378.33 | 376.70 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-07 13:15:00 | 375.90 | 2024-11-13 11:15:00 | 354.37 | TARGET | 21.53 |
| BUY | 2025-01-07 09:15:00 | 449.45 | 2025-01-13 10:15:00 | 419.85 | EXIT_EMA400 | -29.60 |
| BUY | 2025-05-02 09:15:00 | 424.80 | 2025-05-06 14:15:00 | 409.35 | EXIT_EMA400 | -15.45 |
| SELL | 2025-07-18 10:15:00 | 382.20 | 2025-07-31 14:15:00 | 391.95 | EXIT_EMA400 | -9.75 |
| BUY | 2025-11-26 09:15:00 | 390.00 | 2025-12-05 11:15:00 | 381.75 | EXIT_EMA400 | -8.25 |
| BUY | 2025-11-26 11:15:00 | 391.10 | 2025-12-05 11:15:00 | 381.75 | EXIT_EMA400 | -9.35 |
| SELL | 2025-12-26 12:15:00 | 374.00 | 2025-12-30 15:15:00 | 386.25 | EXIT_EMA400 | -12.25 |
| SELL | 2026-03-19 09:15:00 | 368.25 | 2026-04-01 10:15:00 | 377.00 | EXIT_EMA400 | -8.75 |
