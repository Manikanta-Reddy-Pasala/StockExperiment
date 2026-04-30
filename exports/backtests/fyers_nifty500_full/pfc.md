# Power Finance Corporation Ltd. (PFC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 449.65
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 3 |
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| EXIT | 7 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 0 / 7
- **Target hits / EMA400 exits:** 0 / 7
- **Total realized P&L (per unit):** -90.10
- **Avg P&L per closed trade:** -12.87

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 09:15:00 | 483.95 | 509.06 | 509.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 11:15:00 | 482.10 | 508.55 | 508.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 15:15:00 | 473.40 | 471.21 | 484.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-30 09:15:00 | 470.45 | 471.20 | 484.04 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-11 10:15:00 | 486.15 | 466.02 | 478.15 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 10:15:00 | 519.35 | 481.72 | 481.65 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 12:15:00 | 461.10 | 484.30 | 484.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 11:15:00 | 457.95 | 482.97 | 483.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 10:15:00 | 394.45 | 393.17 | 413.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-13 14:15:00 | 388.45 | 394.67 | 410.44 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 398.90 | 395.28 | 409.03 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-21 12:15:00 | 408.95 | 396.23 | 408.84 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 13:15:00 | 432.40 | 413.14 | 413.11 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 401.80 | 413.20 | 413.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 399.45 | 412.93 | 413.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 11:15:00 | 409.95 | 409.30 | 411.15 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-14 12:15:00 | 405.60 | 409.74 | 411.24 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-16 09:15:00 | 421.60 | 409.47 | 411.02 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 09:15:00 | 431.25 | 411.71 | 411.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 09:15:00 | 434.65 | 413.08 | 412.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 405.10 | 414.44 | 413.09 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 11:15:00 | 406.90 | 414.26 | 413.01 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 406.90 | 414.26 | 413.01 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-13 12:15:00 | 404.15 | 414.16 | 412.97 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 15:15:00 | 396.90 | 411.86 | 411.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 395.15 | 411.69 | 411.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 411.50 | 410.50 | 411.20 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 11:15:00 | 424.10 | 411.82 | 411.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 11:15:00 | 426.25 | 412.55 | 412.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 12:15:00 | 414.15 | 414.38 | 413.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-08 09:15:00 | 415.00 | 414.17 | 413.15 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 415.00 | 414.17 | 413.15 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-08 11:15:00 | 412.80 | 414.16 | 413.15 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 13:15:00 | 405.40 | 414.75 | 414.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 14:15:00 | 403.45 | 413.63 | 414.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 401.00 | 400.83 | 405.96 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-12 11:15:00 | 395.95 | 400.68 | 405.66 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-17 10:15:00 | 405.35 | 400.80 | 405.24 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 403.40 | 370.33 | 370.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 11:15:00 | 407.65 | 371.04 | 370.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 397.55 | 401.89 | 391.12 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-05 10:15:00 | 411.90 | 401.62 | 391.41 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-09 09:15:00 | 388.35 | 402.62 | 392.57 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-30 09:15:00 | 470.45 | 2024-11-11 10:15:00 | 486.15 | EXIT_EMA400 | -15.70 |
| SELL | 2025-03-13 14:15:00 | 388.45 | 2025-03-21 12:15:00 | 408.95 | EXIT_EMA400 | -20.50 |
| SELL | 2025-05-14 12:15:00 | 405.60 | 2025-05-16 09:15:00 | 421.60 | EXIT_EMA400 | -16.00 |
| BUY | 2025-06-13 11:15:00 | 406.90 | 2025-06-13 12:15:00 | 404.15 | EXIT_EMA400 | -2.75 |
| BUY | 2025-07-08 09:15:00 | 415.00 | 2025-07-08 11:15:00 | 412.80 | EXIT_EMA400 | -2.20 |
| SELL | 2025-09-12 11:15:00 | 395.95 | 2025-09-17 10:15:00 | 405.35 | EXIT_EMA400 | -9.40 |
| BUY | 2026-03-05 10:15:00 | 411.90 | 2026-03-09 09:15:00 | 388.35 | EXIT_EMA400 | -23.55 |
