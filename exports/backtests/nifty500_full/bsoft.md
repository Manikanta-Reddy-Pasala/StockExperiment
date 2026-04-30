# Birlasoft Ltd. (BSOFT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 369.45
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 37.83
- **Avg P&L per closed trade:** 5.40

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-16 11:15:00 | 704.50 | 753.05 | 753.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-16 12:15:00 | 701.15 | 752.54 | 753.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 15:15:00 | 632.00 | 631.50 | 663.37 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 09:15:00 | 720.15 | 676.70 | 676.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 11:15:00 | 726.25 | 687.09 | 682.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 10:15:00 | 702.65 | 706.16 | 695.76 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 09:15:00 | 602.00 | 686.53 | 686.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 13:15:00 | 586.00 | 682.81 | 684.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 10:15:00 | 633.25 | 632.22 | 652.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-19 10:15:00 | 614.10 | 643.39 | 651.20 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-26 11:15:00 | 593.10 | 572.98 | 591.62 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 11:15:00 | 433.90 | 420.69 | 420.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 11:15:00 | 436.55 | 421.61 | 421.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 426.25 | 426.94 | 424.24 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-15 12:15:00 | 428.75 | 425.99 | 424.03 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 425.05 | 426.96 | 424.72 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-18 12:15:00 | 422.35 | 426.89 | 424.71 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 398.75 | 422.65 | 422.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 396.75 | 422.16 | 422.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 11:15:00 | 420.10 | 417.04 | 419.73 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-30 12:15:00 | 413.40 | 417.01 | 419.70 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 413.40 | 417.01 | 419.70 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-07-30 13:15:00 | 412.45 | 416.96 | 419.66 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 387.75 | 381.26 | 390.93 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-19 10:15:00 | 382.80 | 381.69 | 390.77 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-23 09:15:00 | 380.00 | 358.88 | 371.45 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 14:15:00 | 388.10 | 376.51 | 376.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 399.65 | 379.39 | 378.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 11:15:00 | 422.60 | 424.15 | 409.94 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-07 09:15:00 | 431.45 | 424.12 | 410.28 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 414.00 | 425.63 | 412.47 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-12 10:15:00 | 409.65 | 425.47 | 412.46 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 13:15:00 | 378.40 | 413.95 | 414.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 14:15:00 | 377.65 | 413.59 | 413.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 14:15:00 | 370.50 | 368.72 | 383.86 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 09:15:00 | 361.55 | 369.04 | 381.47 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-15 13:15:00 | 380.95 | 369.27 | 380.92 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-19 10:15:00 | 614.10 | 2024-11-26 11:15:00 | 593.10 | EXIT_EMA400 | 21.00 |
| BUY | 2025-07-15 12:15:00 | 428.75 | 2025-07-18 12:15:00 | 422.35 | EXIT_EMA400 | -6.40 |
| SELL | 2025-07-30 12:15:00 | 413.40 | 2025-07-31 10:15:00 | 394.50 | TARGET | 18.90 |
| SELL | 2025-07-30 13:15:00 | 412.45 | 2025-07-31 15:15:00 | 390.81 | TARGET | 21.64 |
| SELL | 2025-09-19 10:15:00 | 382.80 | 2025-09-26 09:15:00 | 358.90 | TARGET | 23.90 |
| BUY | 2026-01-07 09:15:00 | 431.45 | 2026-01-12 10:15:00 | 409.65 | EXIT_EMA400 | -21.80 |
| SELL | 2026-04-13 09:15:00 | 361.55 | 2026-04-15 13:15:00 | 380.95 | EXIT_EMA400 | -19.40 |
