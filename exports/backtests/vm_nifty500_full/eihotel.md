# EIH Ltd. (EIHOTEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 318.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 6 |
| ENTRY1 | 8 |
| ENTRY2 | 2 |
| EXIT | 8 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 3 / 7
- **Target hits / EMA400 exits:** 3 / 7
- **Total realized P&L (per unit):** 32.66
- **Avg P&L per closed trade:** 3.27

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-27 10:15:00 | 211.85 | 227.54 | 227.59 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 09:15:00 | 239.00 | 227.50 | 227.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-08 10:15:00 | 241.40 | 227.64 | 227.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 12:15:00 | 239.25 | 239.69 | 235.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-28 09:15:00 | 243.00 | 239.12 | 236.02 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 449.05 | 472.11 | 447.41 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-05-27 11:15:00 | 442.60 | 471.59 | 447.39 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 13:15:00 | 432.90 | 438.08 | 438.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 11:15:00 | 431.60 | 437.84 | 437.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 15:15:00 | 425.00 | 424.09 | 429.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-02 11:15:00 | 418.20 | 426.38 | 430.18 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 394.75 | 389.61 | 402.79 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-06 13:15:00 | 387.40 | 389.72 | 402.58 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 388.15 | 380.11 | 390.36 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-10-04 13:15:00 | 390.80 | 380.43 | 390.32 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-17 14:15:00 | 408.10 | 397.04 | 397.00 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-10-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 11:15:00 | 377.80 | 396.97 | 396.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 12:15:00 | 375.15 | 396.75 | 396.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 09:15:00 | 369.95 | 369.03 | 377.81 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 11:15:00 | 427.70 | 382.87 | 382.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 12:15:00 | 434.00 | 383.38 | 383.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 13:15:00 | 402.35 | 402.40 | 394.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-31 12:15:00 | 407.75 | 402.29 | 395.11 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-10 09:15:00 | 399.80 | 409.42 | 400.90 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 13:15:00 | 353.05 | 396.86 | 396.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 09:15:00 | 337.50 | 383.87 | 389.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 10:15:00 | 347.50 | 344.60 | 360.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 11:15:00 | 338.85 | 359.33 | 362.68 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 355.55 | 358.89 | 362.37 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-08 13:15:00 | 362.85 | 358.96 | 362.34 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 13:15:00 | 389.30 | 364.63 | 364.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 14:15:00 | 393.70 | 364.92 | 364.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 14:15:00 | 368.00 | 369.26 | 367.14 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-16 12:15:00 | 371.15 | 366.12 | 365.91 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 369.50 | 369.35 | 367.81 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-27 15:15:00 | 370.00 | 369.36 | 367.82 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-05-28 11:15:00 | 367.55 | 369.35 | 367.84 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 347.60 | 368.06 | 368.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 345.25 | 367.83 | 367.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 13:15:00 | 365.30 | 363.33 | 365.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-27 12:15:00 | 360.95 | 363.42 | 365.43 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-06-30 09:15:00 | 365.40 | 363.31 | 365.33 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 15:15:00 | 382.30 | 366.75 | 366.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 390.15 | 369.67 | 368.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 13:15:00 | 373.85 | 374.59 | 371.39 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-29 09:15:00 | 378.60 | 374.65 | 371.47 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 372.95 | 375.63 | 372.41 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-04 10:15:00 | 372.10 | 375.60 | 372.41 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 14:15:00 | 377.30 | 386.52 | 386.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 09:15:00 | 375.35 | 386.32 | 386.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 380.30 | 379.69 | 382.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-15 14:15:00 | 372.20 | 379.15 | 381.58 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-05 12:15:00 | 365.10 | 339.60 | 353.06 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-12-28 09:15:00 | 243.00 | 2024-01-02 11:15:00 | 263.93 | TARGET | 20.93 |
| SELL | 2024-08-02 11:15:00 | 418.20 | 2024-08-07 09:15:00 | 382.26 | TARGET | 35.94 |
| SELL | 2024-09-06 13:15:00 | 387.40 | 2024-10-04 13:15:00 | 390.80 | EXIT_EMA400 | -3.40 |
| BUY | 2024-12-31 12:15:00 | 407.75 | 2025-01-10 09:15:00 | 399.80 | EXIT_EMA400 | -7.95 |
| SELL | 2025-04-07 11:15:00 | 338.85 | 2025-04-08 13:15:00 | 362.85 | EXIT_EMA400 | -24.00 |
| BUY | 2025-05-16 12:15:00 | 371.15 | 2025-05-28 11:15:00 | 367.55 | EXIT_EMA400 | -3.60 |
| BUY | 2025-05-27 15:15:00 | 370.00 | 2025-05-28 11:15:00 | 367.55 | EXIT_EMA400 | -2.45 |
| SELL | 2025-06-27 12:15:00 | 360.95 | 2025-06-30 09:15:00 | 365.40 | EXIT_EMA400 | -4.45 |
| BUY | 2025-07-29 09:15:00 | 378.60 | 2025-08-04 10:15:00 | 372.10 | EXIT_EMA400 | -6.50 |
| SELL | 2025-12-15 14:15:00 | 372.20 | 2026-01-12 09:15:00 | 344.05 | TARGET | 28.15 |
