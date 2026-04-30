# Railtel Corporation Of India Ltd. (RAILTEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 323.55
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / EMA400 exits:** 3 / 2
- **Total realized P&L (per unit):** 84.42
- **Avg P&L per closed trade:** 16.88

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 09:15:00 | 455.75 | 472.60 | 472.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 11:15:00 | 453.50 | 472.24 | 472.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 10:15:00 | 445.90 | 445.06 | 456.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-17 10:15:00 | 433.65 | 444.60 | 456.12 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 418.60 | 403.01 | 418.61 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-04 09:15:00 | 426.20 | 403.96 | 418.56 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 14:15:00 | 399.15 | 324.35 | 324.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 401.00 | 344.68 | 335.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 11:15:00 | 413.75 | 414.10 | 392.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-15 09:15:00 | 419.25 | 412.81 | 395.36 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-25 09:15:00 | 396.95 | 410.07 | 398.16 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 14:15:00 | 358.45 | 390.19 | 390.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 13:15:00 | 349.25 | 388.05 | 389.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 364.50 | 358.11 | 369.00 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 15:15:00 | 387.70 | 374.99 | 374.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 12:15:00 | 390.75 | 376.02 | 375.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 378.30 | 378.51 | 376.88 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 370.75 | 375.57 | 375.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 13:15:00 | 368.85 | 375.34 | 375.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 14:15:00 | 376.60 | 373.52 | 374.48 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-30 13:15:00 | 369.90 | 373.47 | 374.43 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 368.40 | 365.22 | 369.39 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-18 09:15:00 | 356.95 | 365.16 | 369.23 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-12-23 09:15:00 | 354.75 | 338.94 | 348.97 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 13:15:00 | 356.10 | 355.69 | 355.69 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 14:15:00 | 353.75 | 355.67 | 355.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 349.70 | 355.59 | 355.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 14:15:00 | 346.45 | 346.01 | 350.02 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-29 10:15:00 | 339.30 | 345.92 | 349.92 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 349.15 | 345.71 | 349.67 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-30 11:15:00 | 350.40 | 345.76 | 349.67 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-17 10:15:00 | 433.65 | 2024-11-18 09:15:00 | 366.24 | TARGET | 67.41 |
| BUY | 2025-07-15 09:15:00 | 419.25 | 2025-07-25 09:15:00 | 396.95 | EXIT_EMA400 | -22.30 |
| SELL | 2025-10-30 13:15:00 | 369.90 | 2025-11-06 10:15:00 | 356.31 | TARGET | 13.59 |
| SELL | 2025-11-18 09:15:00 | 356.95 | 2025-12-08 14:15:00 | 320.12 | TARGET | 36.83 |
| SELL | 2026-01-29 10:15:00 | 339.30 | 2026-01-30 11:15:00 | 350.40 | EXIT_EMA400 | -11.10 |
