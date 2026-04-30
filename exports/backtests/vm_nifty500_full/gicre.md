# General Insurance Corporation of India (GICRE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 393.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 18 |
| ALERT1 | 14 |
| ALERT2 | 13 |
| ALERT3 | 8 |
| ENTRY1 | 9 |
| ENTRY2 | 2 |
| EXIT | 9 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 1 / 10
- **Target hits / EMA400 exits:** 1 / 10
- **Total realized P&L (per unit):** -123.25
- **Avg P&L per closed trade:** -11.20

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-04-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 14:15:00 | 329.40 | 347.41 | 347.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 09:15:00 | 325.25 | 347.01 | 347.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-26 14:15:00 | 338.75 | 338.66 | 342.28 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-06 09:15:00 | 332.40 | 340.48 | 342.79 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 332.40 | 340.48 | 342.79 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-05-07 10:15:00 | 325.70 | 339.76 | 342.33 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-05-15 10:15:00 | 342.20 | 334.65 | 339.00 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 09:15:00 | 376.30 | 342.26 | 342.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 10:15:00 | 388.40 | 351.52 | 347.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 391.35 | 392.61 | 378.30 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-26 12:15:00 | 413.30 | 389.38 | 379.04 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 11:15:00 | 388.60 | 397.56 | 385.49 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-08-05 15:15:00 | 385.50 | 397.11 | 385.50 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 11:15:00 | 364.90 | 392.84 | 392.86 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-17 11:15:00 | 395.05 | 392.68 | 392.68 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 10:15:00 | 386.25 | 392.62 | 392.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 10:15:00 | 384.30 | 392.26 | 392.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 13:15:00 | 379.85 | 377.63 | 383.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-07 13:15:00 | 375.90 | 377.77 | 383.45 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-22 09:15:00 | 378.80 | 371.69 | 378.53 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 10:15:00 | 413.00 | 383.37 | 383.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 14:15:00 | 428.50 | 391.91 | 387.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 14:15:00 | 434.15 | 436.25 | 418.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-07 09:15:00 | 448.90 | 436.36 | 418.60 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 425.60 | 439.13 | 422.43 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-13 10:15:00 | 419.85 | 438.93 | 422.41 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 12:15:00 | 383.70 | 417.26 | 417.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 13:15:00 | 381.55 | 416.90 | 417.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 13:15:00 | 388.80 | 387.96 | 397.20 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 09:15:00 | 417.05 | 403.53 | 403.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 423.60 | 405.18 | 404.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 14:15:00 | 405.95 | 406.55 | 405.11 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 11:15:00 | 392.30 | 403.77 | 403.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 13:15:00 | 391.15 | 403.53 | 403.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 09:15:00 | 404.05 | 403.02 | 403.39 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 11:15:00 | 418.20 | 403.84 | 403.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 09:15:00 | 428.80 | 404.62 | 404.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 15:15:00 | 415.05 | 415.35 | 410.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-02 09:15:00 | 425.05 | 415.44 | 410.54 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-06 14:15:00 | 409.35 | 416.83 | 411.72 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 403.75 | 412.73 | 412.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 401.65 | 412.62 | 412.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 386.25 | 385.70 | 394.36 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-18 09:15:00 | 383.25 | 386.25 | 393.79 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 386.40 | 382.57 | 389.80 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-07-31 14:15:00 | 391.95 | 382.82 | 389.64 | Close above EMA400 |

### Cycle 12 — BUY (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 14:15:00 | 383.50 | 378.25 | 378.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 10:15:00 | 385.35 | 378.43 | 378.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 377.85 | 379.05 | 378.66 | EMA200 retest candle locked |

### Cycle 13 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 371.70 | 378.32 | 378.33 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 12:15:00 | 382.05 | 378.36 | 378.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 13:15:00 | 383.00 | 378.41 | 378.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 381.40 | 383.08 | 381.02 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-26 09:15:00 | 390.00 | 382.67 | 381.01 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 390.00 | 382.67 | 381.01 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-11-26 11:15:00 | 391.10 | 382.83 | 381.11 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 383.95 | 384.30 | 382.31 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-05 11:15:00 | 381.75 | 384.27 | 382.31 | Close below EMA400 |

### Cycle 15 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 364.05 | 380.86 | 380.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 14:15:00 | 362.40 | 380.01 | 380.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 377.35 | 377.33 | 378.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-26 12:15:00 | 374.00 | 377.29 | 378.91 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 378.05 | 376.38 | 378.31 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-30 15:15:00 | 386.00 | 376.48 | 378.34 | Close above EMA400 |

### Cycle 16 — BUY (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 14:15:00 | 393.00 | 376.03 | 376.03 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2026-03-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 12:15:00 | 364.70 | 377.71 | 377.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 362.50 | 375.12 | 376.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 10:15:00 | 375.10 | 373.30 | 375.26 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-19 09:15:00 | 368.25 | 373.25 | 375.17 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 371.35 | 368.89 | 372.32 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-01 10:15:00 | 377.00 | 368.97 | 372.34 | Close above EMA400 |

### Cycle 18 — BUY (started 2026-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 13:15:00 | 397.50 | 375.01 | 374.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 417.00 | 378.33 | 376.68 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-05-06 09:15:00 | 332.40 | 2024-05-15 10:15:00 | 342.20 | EXIT_EMA400 | -9.80 |
| SELL | 2024-05-07 10:15:00 | 325.70 | 2024-05-15 10:15:00 | 342.20 | EXIT_EMA400 | -16.50 |
| BUY | 2024-07-26 12:15:00 | 413.30 | 2024-08-05 15:15:00 | 385.50 | EXIT_EMA400 | -27.80 |
| SELL | 2024-11-07 13:15:00 | 375.90 | 2024-11-13 11:15:00 | 353.25 | TARGET | 22.65 |
| BUY | 2025-01-07 09:15:00 | 448.90 | 2025-01-13 10:15:00 | 419.85 | EXIT_EMA400 | -29.05 |
| BUY | 2025-05-02 09:15:00 | 425.05 | 2025-05-06 14:15:00 | 409.35 | EXIT_EMA400 | -15.70 |
| SELL | 2025-07-18 09:15:00 | 383.25 | 2025-07-31 14:15:00 | 391.95 | EXIT_EMA400 | -8.70 |
| BUY | 2025-11-26 09:15:00 | 390.00 | 2025-12-05 11:15:00 | 381.75 | EXIT_EMA400 | -8.25 |
| BUY | 2025-11-26 11:15:00 | 391.10 | 2025-12-05 11:15:00 | 381.75 | EXIT_EMA400 | -9.35 |
| SELL | 2025-12-26 12:15:00 | 374.00 | 2025-12-30 15:15:00 | 386.00 | EXIT_EMA400 | -12.00 |
| SELL | 2026-03-19 09:15:00 | 368.25 | 2026-04-01 10:15:00 | 377.00 | EXIT_EMA400 | -8.75 |
