# JK Tyre & Industries Ltd. (JKTYRE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 406.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 1
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** 52.81
- **Avg P&L per closed trade:** 10.56

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 15:15:00 | 422.00 | 446.03 | 446.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 10:15:00 | 420.80 | 445.56 | 445.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 11:15:00 | 429.60 | 425.40 | 433.43 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-03 09:15:00 | 418.05 | 425.50 | 433.02 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-05-22 09:15:00 | 428.85 | 408.92 | 420.49 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 09:15:00 | 461.75 | 415.80 | 415.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 12:15:00 | 465.15 | 426.04 | 421.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 09:15:00 | 433.50 | 438.25 | 429.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-24 12:15:00 | 443.00 | 438.02 | 429.51 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-08-02 09:15:00 | 430.70 | 439.95 | 432.32 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 10:15:00 | 391.70 | 426.91 | 426.93 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 11:15:00 | 423.55 | 421.49 | 421.49 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 11:15:00 | 418.45 | 421.46 | 421.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 14:15:00 | 416.65 | 421.35 | 421.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 09:15:00 | 421.40 | 421.32 | 421.40 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 13:15:00 | 434.00 | 421.57 | 421.53 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 388.85 | 421.42 | 421.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 15:15:00 | 384.90 | 406.59 | 412.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 382.55 | 381.94 | 393.41 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 13:15:00 | 372.00 | 390.67 | 393.43 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-15 11:15:00 | 300.70 | 282.93 | 298.51 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 332.75 | 305.76 | 305.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 334.50 | 306.05 | 305.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 356.75 | 360.50 | 343.27 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-07 09:15:00 | 370.70 | 359.55 | 349.55 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-22 11:15:00 | 354.75 | 364.09 | 355.60 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 321.10 | 350.34 | 350.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 318.55 | 348.63 | 349.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 347.60 | 332.48 | 338.78 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 15:15:00 | 367.15 | 343.13 | 343.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 10:15:00 | 370.10 | 343.65 | 343.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 357.40 | 358.64 | 352.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-26 15:15:00 | 361.45 | 358.67 | 352.26 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-02-27 10:15:00 | 514.60 | 539.70 | 516.11 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 11:15:00 | 426.15 | 499.09 | 499.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 418.50 | 495.55 | 497.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 12:15:00 | 432.40 | 432.29 | 455.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 09:15:00 | 419.80 | 432.20 | 455.13 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-05-03 09:15:00 | 418.05 | 2024-05-22 09:15:00 | 428.85 | EXIT_EMA400 | -10.80 |
| BUY | 2024-07-24 12:15:00 | 443.00 | 2024-08-02 09:15:00 | 430.70 | EXIT_EMA400 | -12.30 |
| SELL | 2025-01-06 13:15:00 | 372.00 | 2025-01-28 10:15:00 | 307.70 | TARGET | 64.30 |
| BUY | 2025-07-07 09:15:00 | 370.70 | 2025-07-22 11:15:00 | 354.75 | EXIT_EMA400 | -15.95 |
| BUY | 2025-09-26 15:15:00 | 361.45 | 2025-10-08 11:15:00 | 389.01 | TARGET | 27.56 |
