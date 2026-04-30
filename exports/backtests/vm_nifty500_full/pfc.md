# Power Finance Corporation Ltd. (PFC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 448.40
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
- **Total realized P&L (per unit):** -96.30
- **Avg P&L per closed trade:** -13.76

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 15:15:00 | 489.95 | 509.33 | 509.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 09:15:00 | 483.70 | 509.07 | 509.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 15:15:00 | 474.00 | 471.20 | 484.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-30 09:15:00 | 470.45 | 471.19 | 484.04 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-11 10:15:00 | 486.15 | 466.18 | 478.36 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 11:15:00 | 514.80 | 482.08 | 481.92 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 12:15:00 | 460.95 | 484.32 | 484.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 11:15:00 | 458.00 | 482.99 | 483.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 10:15:00 | 394.50 | 393.45 | 414.38 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-13 14:15:00 | 388.40 | 394.83 | 410.90 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 398.90 | 395.40 | 409.43 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-24 09:15:00 | 415.30 | 396.88 | 409.24 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 15:15:00 | 434.20 | 413.58 | 413.51 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 402.85 | 413.43 | 413.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 10:15:00 | 402.20 | 413.32 | 413.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 11:15:00 | 409.75 | 409.31 | 411.28 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-14 12:15:00 | 405.60 | 409.75 | 411.36 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-16 09:15:00 | 421.60 | 409.47 | 411.13 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 09:15:00 | 431.25 | 411.70 | 411.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 09:15:00 | 434.60 | 413.07 | 412.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 405.10 | 414.41 | 413.12 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 11:15:00 | 406.90 | 414.23 | 413.05 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 406.90 | 414.23 | 413.05 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-13 12:15:00 | 404.15 | 414.13 | 413.00 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 14:15:00 | 397.00 | 411.99 | 412.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 393.50 | 411.49 | 411.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 411.50 | 410.48 | 411.23 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 11:15:00 | 424.10 | 411.81 | 411.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 11:15:00 | 426.20 | 412.55 | 412.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 12:15:00 | 414.00 | 414.38 | 413.22 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-08 09:15:00 | 414.95 | 414.17 | 413.18 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 414.95 | 414.17 | 413.18 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-08 11:15:00 | 412.80 | 414.16 | 413.18 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 13:15:00 | 405.60 | 414.75 | 414.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 14:15:00 | 403.40 | 413.63 | 414.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 401.00 | 400.81 | 405.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-12 11:15:00 | 395.95 | 400.66 | 405.65 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-17 10:15:00 | 405.35 | 400.78 | 405.23 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 12:15:00 | 409.15 | 370.60 | 370.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 412.00 | 371.02 | 370.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 397.60 | 401.69 | 390.94 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-05 10:15:00 | 411.90 | 401.48 | 391.25 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-09 09:15:00 | 388.50 | 402.48 | 392.41 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-30 09:15:00 | 470.45 | 2024-11-11 10:15:00 | 486.15 | EXIT_EMA400 | -15.70 |
| SELL | 2025-03-13 14:15:00 | 388.40 | 2025-03-24 09:15:00 | 415.30 | EXIT_EMA400 | -26.90 |
| SELL | 2025-05-14 12:15:00 | 405.60 | 2025-05-16 09:15:00 | 421.60 | EXIT_EMA400 | -16.00 |
| BUY | 2025-06-13 11:15:00 | 406.90 | 2025-06-13 12:15:00 | 404.15 | EXIT_EMA400 | -2.75 |
| BUY | 2025-07-08 09:15:00 | 414.95 | 2025-07-08 11:15:00 | 412.80 | EXIT_EMA400 | -2.15 |
| SELL | 2025-09-12 11:15:00 | 395.95 | 2025-09-17 10:15:00 | 405.35 | EXIT_EMA400 | -9.40 |
| BUY | 2026-03-05 10:15:00 | 411.90 | 2026-03-09 09:15:00 | 388.50 | EXIT_EMA400 | -23.40 |
