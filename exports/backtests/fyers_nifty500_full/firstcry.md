# Brainbees Solutions Ltd. (FIRSTCRY.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-08-13 09:15:00 → 2026-04-30 15:15:00 (2965 bars)
- **Last close:** 238.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 17.38
- **Avg P&L per closed trade:** 4.34

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 12:15:00 | 650.80 | 616.16 | 616.06 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 11:15:00 | 572.05 | 615.82 | 615.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 12:15:00 | 571.00 | 615.37 | 615.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 394.10 | 390.91 | 432.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-04 09:15:00 | 355.25 | 385.27 | 424.03 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 360.15 | 348.36 | 369.84 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-05-23 14:15:00 | 373.50 | 349.13 | 369.70 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 12:15:00 | 380.40 | 372.19 | 372.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 13:15:00 | 382.35 | 372.29 | 372.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 10:15:00 | 372.15 | 373.35 | 372.78 | EMA200 retest candle locked |

### Cycle 4 — SELL (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 13:15:00 | 362.25 | 372.19 | 372.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 358.25 | 371.86 | 372.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 364.10 | 363.73 | 367.46 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 15:15:00 | 389.00 | 369.21 | 369.15 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 14:15:00 | 355.70 | 369.18 | 369.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 352.35 | 367.74 | 368.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 373.75 | 367.51 | 368.35 | EMA200 retest candle locked |

### Cycle 7 — BUY (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 15:15:00 | 399.90 | 369.23 | 369.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 413.20 | 369.67 | 369.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 379.30 | 382.09 | 377.05 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-23 11:15:00 | 385.75 | 382.03 | 377.12 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 385.75 | 382.03 | 377.12 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-23 13:15:00 | 386.90 | 382.12 | 377.21 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 377.85 | 382.16 | 377.59 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-26 09:15:00 | 372.65 | 382.04 | 377.57 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 10:15:00 | 362.70 | 374.62 | 374.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 11:15:00 | 359.20 | 371.72 | 373.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 12:15:00 | 298.60 | 298.47 | 315.14 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-06 09:15:00 | 293.80 | 298.44 | 314.79 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-03-20 10:15:00 | 249.86 | 229.11 | 249.19 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-04 09:15:00 | 355.25 | 2025-05-23 14:15:00 | 373.50 | EXIT_EMA400 | -18.25 |
| BUY | 2025-09-23 11:15:00 | 385.75 | 2025-09-26 09:15:00 | 372.65 | EXIT_EMA400 | -13.10 |
| BUY | 2025-09-23 13:15:00 | 386.90 | 2025-09-26 09:15:00 | 372.65 | EXIT_EMA400 | -14.25 |
| SELL | 2026-01-06 09:15:00 | 293.80 | 2026-02-17 09:15:00 | 230.82 | TARGET | 62.98 |
