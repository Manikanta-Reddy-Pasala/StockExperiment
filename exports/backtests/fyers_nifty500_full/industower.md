# Indus Towers Ltd. (INDUSTOWER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 410.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 0.44
- **Avg P&L per closed trade:** 0.11

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 12:15:00 | 356.65 | 405.01 | 405.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 09:15:00 | 354.15 | 388.46 | 395.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 09:15:00 | 348.65 | 345.76 | 363.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-26 10:15:00 | 344.35 | 345.75 | 363.56 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 359.30 | 347.19 | 360.59 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-05 09:15:00 | 362.60 | 347.34 | 360.60 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 10:15:00 | 363.65 | 343.92 | 343.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 11:15:00 | 366.00 | 344.14 | 343.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 09:15:00 | 385.30 | 385.55 | 373.96 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-28 09:15:00 | 392.35 | 384.77 | 375.06 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 379.55 | 386.41 | 379.36 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-13 10:15:00 | 377.60 | 386.33 | 379.35 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 13:15:00 | 346.15 | 391.90 | 391.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 345.10 | 390.54 | 391.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 351.65 | 348.74 | 361.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-10 14:15:00 | 349.10 | 348.80 | 361.48 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 360.50 | 349.27 | 361.16 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-12 11:15:00 | 361.55 | 349.50 | 361.16 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 15:15:00 | 382.75 | 359.33 | 359.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 09:15:00 | 390.90 | 359.65 | 359.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 10:15:00 | 422.65 | 422.87 | 409.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-28 10:15:00 | 426.25 | 420.98 | 410.95 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-04 09:15:00 | 437.30 | 452.99 | 438.36 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 09:15:00 | 418.00 | 434.21 | 434.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 12:15:00 | 413.75 | 433.70 | 434.00 | Break + close below crossover candle low |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-26 10:15:00 | 344.35 | 2024-12-05 09:15:00 | 362.60 | EXIT_EMA400 | -18.25 |
| BUY | 2025-05-28 09:15:00 | 392.35 | 2025-06-13 10:15:00 | 377.60 | EXIT_EMA400 | -14.75 |
| SELL | 2025-09-10 14:15:00 | 349.10 | 2025-09-12 11:15:00 | 361.55 | EXIT_EMA400 | -12.45 |
| BUY | 2026-01-28 10:15:00 | 426.25 | 2026-02-12 09:15:00 | 472.14 | TARGET | 45.89 |
