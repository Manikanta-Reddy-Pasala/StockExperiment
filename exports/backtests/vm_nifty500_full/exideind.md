# Exide Industries Ltd. (EXIDEIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 360.55
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** 77.29
- **Avg P&L per closed trade:** 9.66

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 13:15:00 | 305.25 | 315.25 | 315.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 15:15:00 | 304.40 | 315.04 | 315.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-02 14:15:00 | 314.30 | 314.09 | 314.63 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 10:15:00 | 336.20 | 315.17 | 315.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 11:15:00 | 360.15 | 315.62 | 315.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 461.65 | 463.23 | 425.71 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-05 14:15:00 | 491.30 | 464.35 | 428.28 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 521.35 | 547.57 | 521.14 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-31 12:15:00 | 526.35 | 547.11 | 521.18 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-08-01 10:15:00 | 519.40 | 545.95 | 521.23 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 12:15:00 | 496.65 | 509.04 | 509.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 09:15:00 | 491.65 | 508.42 | 508.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 10:15:00 | 487.60 | 484.60 | 493.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-07 12:15:00 | 478.75 | 489.42 | 494.49 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 490.20 | 489.15 | 494.21 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-10-09 09:15:00 | 500.65 | 489.42 | 494.22 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 13:15:00 | 521.85 | 498.45 | 498.36 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-10-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 11:15:00 | 471.20 | 498.26 | 498.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 12:15:00 | 469.25 | 497.97 | 498.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 455.35 | 448.15 | 465.02 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-19 09:15:00 | 439.75 | 453.87 | 461.76 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 367.70 | 357.62 | 373.03 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-26 10:15:00 | 365.00 | 357.69 | 372.99 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-04-02 12:15:00 | 372.20 | 358.91 | 371.50 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 13:15:00 | 391.80 | 372.98 | 372.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 395.25 | 382.30 | 378.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 10:15:00 | 386.55 | 386.87 | 381.62 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-27 09:15:00 | 392.50 | 384.98 | 382.00 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 384.20 | 385.51 | 382.60 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-02 11:15:00 | 385.15 | 385.50 | 382.61 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 383.00 | 385.47 | 382.62 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-03 10:15:00 | 381.90 | 385.36 | 382.62 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 14:15:00 | 375.35 | 383.58 | 383.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 373.85 | 383.41 | 383.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 13:15:00 | 383.50 | 382.32 | 382.92 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 396.60 | 383.59 | 383.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 11:15:00 | 397.35 | 383.72 | 383.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 12:15:00 | 408.10 | 408.49 | 399.73 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 13:15:00 | 379.60 | 397.06 | 397.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 378.20 | 392.44 | 394.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 380.35 | 379.93 | 385.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-03 09:15:00 | 374.40 | 379.81 | 385.42 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-10 10:15:00 | 322.70 | 309.82 | 322.16 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 15:15:00 | 363.45 | 329.19 | 329.10 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-06-05 14:15:00 | 491.30 | 2024-08-01 10:15:00 | 519.40 | EXIT_EMA400 | 28.10 |
| BUY | 2024-07-31 12:15:00 | 526.35 | 2024-08-01 10:15:00 | 519.40 | EXIT_EMA400 | -6.95 |
| SELL | 2024-10-07 12:15:00 | 478.75 | 2024-10-09 09:15:00 | 500.65 | EXIT_EMA400 | -21.90 |
| SELL | 2024-12-19 09:15:00 | 439.75 | 2025-01-13 14:15:00 | 373.72 | TARGET | 66.03 |
| SELL | 2025-03-26 10:15:00 | 365.00 | 2025-04-02 12:15:00 | 372.20 | EXIT_EMA400 | -7.20 |
| BUY | 2025-06-27 09:15:00 | 392.50 | 2025-07-03 10:15:00 | 381.90 | EXIT_EMA400 | -10.60 |
| BUY | 2025-07-02 11:15:00 | 385.15 | 2025-07-03 10:15:00 | 381.90 | EXIT_EMA400 | -3.25 |
| SELL | 2025-12-03 09:15:00 | 374.40 | 2026-01-12 09:15:00 | 341.33 | TARGET | 33.07 |
