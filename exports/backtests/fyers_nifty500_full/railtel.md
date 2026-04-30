# Railtel Corporation Of India Ltd. (RAILTEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 321.79
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 3 |
| EXIT | 3 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / EMA400 exits:** 4 / 2
- **Total realized P&L (per unit):** 107.42
- **Avg P&L per closed trade:** 17.90

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 445.50 | 471.03 | 471.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 444.45 | 470.76 | 470.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 10:15:00 | 445.90 | 445.08 | 456.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-17 10:15:00 | 433.85 | 444.62 | 455.61 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-03 09:15:00 | 418.50 | 402.97 | 418.33 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 14:15:00 | 399.15 | 324.35 | 324.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 401.00 | 344.68 | 335.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 11:15:00 | 413.75 | 414.08 | 392.05 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-15 09:15:00 | 419.05 | 412.79 | 395.34 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 400.20 | 411.37 | 397.91 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-23 15:15:00 | 404.05 | 410.78 | 398.01 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-25 09:15:00 | 396.95 | 410.06 | 398.14 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 14:15:00 | 358.45 | 390.17 | 390.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 13:15:00 | 349.25 | 388.04 | 389.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 364.50 | 358.11 | 368.99 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 15:15:00 | 387.70 | 375.02 | 374.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 12:15:00 | 390.75 | 376.06 | 375.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 378.30 | 378.53 | 376.89 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 370.75 | 375.60 | 375.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 13:15:00 | 368.85 | 375.36 | 375.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 14:15:00 | 376.60 | 373.53 | 374.49 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-30 13:15:00 | 369.90 | 373.49 | 374.44 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 369.80 | 373.05 | 374.17 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-04 09:15:00 | 367.05 | 372.81 | 374.01 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 368.30 | 365.22 | 369.40 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-18 09:15:00 | 356.95 | 365.16 | 369.23 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-12-23 09:15:00 | 354.75 | 338.92 | 348.97 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-17 10:15:00 | 433.85 | 2024-11-18 09:15:00 | 368.57 | TARGET | 65.28 |
| BUY | 2025-07-15 09:15:00 | 419.05 | 2025-07-25 09:15:00 | 396.95 | EXIT_EMA400 | -22.10 |
| BUY | 2025-07-23 15:15:00 | 404.05 | 2025-07-25 09:15:00 | 396.95 | EXIT_EMA400 | -7.10 |
| SELL | 2025-10-30 13:15:00 | 369.90 | 2025-11-06 10:15:00 | 356.28 | TARGET | 13.62 |
| SELL | 2025-11-04 09:15:00 | 367.05 | 2025-11-11 09:15:00 | 346.17 | TARGET | 20.88 |
| SELL | 2025-11-18 09:15:00 | 356.95 | 2025-12-08 13:15:00 | 320.11 | TARGET | 36.84 |
