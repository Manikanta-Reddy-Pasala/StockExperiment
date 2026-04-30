# Indus Towers Ltd. (INDUSTOWER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 409.95
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
- **Total realized P&L (per unit):** -2.01
- **Avg P&L per closed trade:** -0.50

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 09:15:00 | 365.05 | 403.34 | 403.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 14:15:00 | 357.05 | 389.13 | 395.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 09:15:00 | 348.55 | 345.98 | 363.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-26 10:15:00 | 344.35 | 345.97 | 363.57 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 359.70 | 347.32 | 360.59 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-05 09:15:00 | 362.60 | 347.47 | 360.60 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 10:15:00 | 363.90 | 343.95 | 343.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 11:15:00 | 366.00 | 344.17 | 344.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 09:15:00 | 385.20 | 385.56 | 373.99 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-28 09:15:00 | 392.30 | 384.77 | 375.08 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 379.55 | 386.47 | 379.41 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-13 10:15:00 | 377.60 | 386.38 | 379.40 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 13:15:00 | 346.15 | 391.94 | 391.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 14:15:00 | 345.75 | 391.48 | 391.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 13:15:00 | 348.85 | 348.73 | 361.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-09 14:15:00 | 346.60 | 348.71 | 361.89 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 360.50 | 349.26 | 361.16 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-12 11:15:00 | 361.55 | 349.48 | 361.16 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 15:15:00 | 382.75 | 359.34 | 359.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 09:15:00 | 390.90 | 359.66 | 359.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 10:15:00 | 422.65 | 422.89 | 409.93 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-28 10:15:00 | 426.25 | 420.97 | 410.95 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-04 09:15:00 | 437.25 | 452.87 | 438.05 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 10:15:00 | 419.50 | 434.04 | 434.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 12:15:00 | 413.75 | 433.67 | 433.87 | Break + close below crossover candle low |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-26 10:15:00 | 344.35 | 2024-12-05 09:15:00 | 362.60 | EXIT_EMA400 | -18.25 |
| BUY | 2025-05-28 09:15:00 | 392.30 | 2025-06-13 10:15:00 | 377.60 | EXIT_EMA400 | -14.70 |
| SELL | 2025-09-09 14:15:00 | 346.60 | 2025-09-12 11:15:00 | 361.55 | EXIT_EMA400 | -14.95 |
| BUY | 2026-01-28 10:15:00 | 426.25 | 2026-02-12 09:15:00 | 472.14 | TARGET | 45.89 |
