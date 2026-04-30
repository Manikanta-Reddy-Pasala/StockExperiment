# JK Tyre & Industries Ltd. (JKTYRE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 410.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 1
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** -1.27
- **Avg P&L per closed trade:** -0.32

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 11:15:00 | 407.05 | 417.60 | 417.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 14:15:00 | 406.25 | 417.28 | 417.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 09:15:00 | 418.25 | 413.73 | 415.53 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 13:15:00 | 440.60 | 416.79 | 416.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 09:15:00 | 448.60 | 417.60 | 417.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 10:15:00 | 419.70 | 420.87 | 418.97 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-19 15:15:00 | 424.90 | 420.87 | 419.02 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-09-24 09:15:00 | 417.95 | 421.46 | 419.46 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 13:15:00 | 398.70 | 418.68 | 418.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 392.00 | 412.16 | 415.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 400.15 | 400.11 | 407.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-01 18:15:00 | 390.55 | 400.02 | 407.10 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-04 09:15:00 | 396.60 | 382.30 | 391.64 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 332.10 | 305.46 | 305.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 335.00 | 306.87 | 306.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 356.75 | 360.47 | 343.16 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-07 09:15:00 | 370.70 | 359.53 | 349.49 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-22 11:15:00 | 354.75 | 364.09 | 355.57 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 321.10 | 350.35 | 350.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 318.55 | 348.64 | 349.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 347.55 | 332.50 | 338.78 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 15:15:00 | 369.00 | 343.16 | 343.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 10:15:00 | 370.10 | 343.68 | 343.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 357.40 | 358.68 | 352.21 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-29 09:15:00 | 361.55 | 358.72 | 352.32 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-02-27 10:15:00 | 514.70 | 539.83 | 516.60 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 11:15:00 | 426.15 | 499.17 | 499.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 418.50 | 495.62 | 497.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 13:15:00 | 433.70 | 432.33 | 455.73 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 09:15:00 | 419.80 | 432.22 | 455.33 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-09-19 15:15:00 | 424.90 | 2024-09-24 09:15:00 | 417.95 | EXIT_EMA400 | -6.95 |
| SELL | 2024-11-01 18:15:00 | 390.55 | 2024-12-04 09:15:00 | 396.60 | EXIT_EMA400 | -6.05 |
| BUY | 2025-07-07 09:15:00 | 370.70 | 2025-07-22 11:15:00 | 354.75 | EXIT_EMA400 | -15.95 |
| BUY | 2025-09-29 09:15:00 | 361.55 | 2025-10-08 11:15:00 | 389.23 | TARGET | 27.68 |
