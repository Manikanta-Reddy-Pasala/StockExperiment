# Brainbees Solutions Ltd. (FIRSTCRY.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-08-13 09:15:00 → 2026-04-30 15:30:00 (2946 bars)
- **Last close:** 239.21
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / EMA400 exits:** 3 / 3
- **Total realized P&L (per unit):** 42.26
- **Avg P&L per closed trade:** 7.04

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 12:15:00 | 650.80 | 616.22 | 616.13 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 11:15:00 | 572.00 | 615.89 | 616.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 12:15:00 | 571.00 | 615.44 | 615.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 394.10 | 391.06 | 433.50 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-04 09:15:00 | 355.25 | 385.44 | 424.59 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-23 14:15:00 | 373.70 | 349.19 | 369.91 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 382.35 | 372.29 | 372.27 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 12:15:00 | 362.40 | 372.30 | 372.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 358.25 | 371.87 | 372.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 364.10 | 363.74 | 367.50 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 15:15:00 | 389.10 | 369.22 | 369.17 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 13:15:00 | 359.10 | 369.33 | 369.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-01 14:15:00 | 355.70 | 369.19 | 369.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-04 09:15:00 | 375.30 | 368.06 | 368.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-04 10:15:00 | 365.30 | 368.03 | 368.66 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 365.30 | 368.03 | 368.66 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-04 11:15:00 | 363.60 | 367.99 | 368.63 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-05 09:15:00 | 373.75 | 367.51 | 368.37 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 15:15:00 | 399.85 | 369.24 | 369.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 413.20 | 369.67 | 369.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 379.50 | 382.07 | 377.05 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-23 11:15:00 | 385.85 | 382.01 | 377.12 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 385.85 | 382.01 | 377.12 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-23 13:15:00 | 386.90 | 382.10 | 377.21 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 377.85 | 382.14 | 377.59 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-26 09:15:00 | 372.65 | 382.02 | 377.57 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 10:15:00 | 362.70 | 374.64 | 374.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 11:15:00 | 359.20 | 371.74 | 373.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 12:15:00 | 298.60 | 298.48 | 315.15 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-06 09:15:00 | 293.80 | 298.45 | 314.80 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-03-20 10:15:00 | 249.90 | 229.17 | 249.47 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-04 09:15:00 | 355.25 | 2025-05-23 14:15:00 | 373.70 | EXIT_EMA400 | -18.45 |
| SELL | 2025-09-04 10:15:00 | 365.30 | 2025-09-04 13:15:00 | 355.23 | TARGET | 10.07 |
| SELL | 2025-09-04 11:15:00 | 363.60 | 2025-09-04 14:15:00 | 348.51 | TARGET | 15.09 |
| BUY | 2025-09-23 11:15:00 | 385.85 | 2025-09-26 09:15:00 | 372.65 | EXIT_EMA400 | -13.20 |
| BUY | 2025-09-23 13:15:00 | 386.90 | 2025-09-26 09:15:00 | 372.65 | EXIT_EMA400 | -14.25 |
| SELL | 2026-01-06 09:15:00 | 293.80 | 2026-02-17 09:15:00 | 230.80 | TARGET | 63.00 |
