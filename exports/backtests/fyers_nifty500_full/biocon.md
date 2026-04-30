# Biocon Ltd. (BIOCON.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 358.65
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 4 |
| ENTRY1 | 7 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 2
- **Winners / losers:** 0 / 7
- **Target hits / EMA400 exits:** 0 / 7
- **Total realized P&L (per unit):** -78.75
- **Avg P&L per closed trade:** -11.25

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 13:15:00 | 336.90 | 354.29 | 354.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 334.35 | 353.75 | 354.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 09:15:00 | 335.95 | 333.90 | 341.79 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-18 14:15:00 | 325.50 | 334.90 | 341.18 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 338.60 | 333.68 | 339.63 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-26 10:15:00 | 342.05 | 333.76 | 339.64 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 15:15:00 | 373.50 | 344.27 | 344.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 14:15:00 | 380.15 | 345.97 | 345.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 350.30 | 353.97 | 349.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-18 10:15:00 | 356.90 | 353.61 | 350.05 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 350.80 | 353.61 | 350.09 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-18 13:15:00 | 349.75 | 353.57 | 350.08 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 13:15:00 | 334.75 | 363.00 | 363.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 09:15:00 | 328.00 | 362.14 | 362.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 13:15:00 | 337.00 | 336.69 | 345.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-04 09:15:00 | 324.95 | 340.70 | 345.29 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 334.05 | 332.61 | 338.98 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-24 09:15:00 | 327.95 | 332.79 | 338.85 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-05-05 13:15:00 | 335.65 | 328.81 | 335.36 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 14:15:00 | 352.75 | 336.63 | 336.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-13 15:15:00 | 357.55 | 337.90 | 337.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 09:15:00 | 379.30 | 380.66 | 367.53 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 361.10 | 362.59 | 362.60 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 13:15:00 | 366.50 | 362.63 | 362.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 14:15:00 | 368.35 | 362.69 | 362.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 12:15:00 | 362.85 | 363.30 | 362.96 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 11:15:00 | 356.45 | 362.64 | 362.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 355.10 | 362.45 | 362.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 11:15:00 | 355.75 | 355.03 | 358.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-13 10:15:00 | 348.45 | 354.84 | 357.92 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-16 09:15:00 | 358.00 | 354.28 | 357.32 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 375.40 | 359.27 | 359.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 14:15:00 | 380.40 | 362.28 | 360.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 11:15:00 | 388.75 | 391.04 | 380.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-19 09:15:00 | 399.40 | 388.65 | 382.55 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-06 14:15:00 | 384.95 | 391.53 | 386.47 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 10:15:00 | 371.00 | 383.29 | 383.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-22 11:15:00 | 369.95 | 383.16 | 383.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 376.55 | 376.42 | 379.38 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-03 12:15:00 | 372.90 | 376.38 | 379.35 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-12 09:15:00 | 385.55 | 374.18 | 377.49 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 396.20 | 379.49 | 379.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 10:15:00 | 400.90 | 383.45 | 381.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 10:15:00 | 384.05 | 384.84 | 382.52 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 15:15:00 | 369.10 | 380.88 | 380.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 362.85 | 380.07 | 380.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 364.50 | 363.29 | 370.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-22 09:15:00 | 358.50 | 363.18 | 369.78 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 362.95 | 361.66 | 367.88 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-29 13:15:00 | 362.55 | 361.78 | 367.82 | Sell entry 2 (retest2 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-18 14:15:00 | 325.50 | 2024-11-26 10:15:00 | 342.05 | EXIT_EMA400 | -16.55 |
| BUY | 2024-12-18 10:15:00 | 356.90 | 2024-12-18 13:15:00 | 349.75 | EXIT_EMA400 | -7.15 |
| SELL | 2025-04-04 09:15:00 | 324.95 | 2025-05-05 13:15:00 | 335.65 | EXIT_EMA400 | -10.70 |
| SELL | 2025-04-24 09:15:00 | 327.95 | 2025-05-05 13:15:00 | 335.65 | EXIT_EMA400 | -7.70 |
| SELL | 2025-10-13 10:15:00 | 348.45 | 2025-10-16 09:15:00 | 358.00 | EXIT_EMA400 | -9.55 |
| BUY | 2025-12-19 09:15:00 | 399.40 | 2026-01-06 14:15:00 | 384.95 | EXIT_EMA400 | -14.45 |
| SELL | 2026-02-03 12:15:00 | 372.90 | 2026-02-12 09:15:00 | 385.55 | EXIT_EMA400 | -12.65 |
