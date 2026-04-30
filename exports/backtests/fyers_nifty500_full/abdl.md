# Allied Blenders and Distillers Ltd. (ABDL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-07-02 09:15:00 → 2026-04-30 15:15:00 (3168 bars)
- **Last close:** 533.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 4 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 35.31
- **Avg P&L per closed trade:** 5.04

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 10:15:00 | 303.60 | 329.28 | 329.37 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 14:15:00 | 352.50 | 327.14 | 327.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 12:15:00 | 364.00 | 330.02 | 328.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 11:15:00 | 396.00 | 400.66 | 376.75 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-15 10:15:00 | 404.95 | 399.76 | 377.78 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-27 09:15:00 | 377.30 | 400.52 | 383.58 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 10:15:00 | 325.55 | 378.07 | 378.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 316.10 | 362.55 | 369.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 09:15:00 | 326.10 | 321.46 | 335.66 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-15 11:15:00 | 296.50 | 321.28 | 335.43 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 329.35 | 320.94 | 333.03 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-24 13:15:00 | 326.65 | 321.93 | 332.89 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 326.20 | 319.06 | 328.41 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-05-08 12:15:00 | 322.95 | 319.17 | 328.37 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-05-09 11:15:00 | 330.50 | 319.27 | 328.15 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 10:15:00 | 401.00 | 335.89 | 335.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 405.00 | 342.92 | 339.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 498.35 | 498.73 | 473.45 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-08 10:15:00 | 513.75 | 498.53 | 475.86 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 603.20 | 627.03 | 599.97 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-09 11:15:00 | 621.80 | 626.81 | 600.13 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-12-18 09:15:00 | 603.00 | 624.34 | 604.36 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 507.60 | 597.29 | 597.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 10:15:00 | 504.45 | 596.37 | 597.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 510.80 | 509.40 | 541.40 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-04 11:15:00 | 503.45 | 509.41 | 540.00 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-12 09:15:00 | 536.35 | 509.87 | 534.69 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 13:15:00 | 564.80 | 489.93 | 489.79 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-01-15 10:15:00 | 404.95 | 2025-01-27 09:15:00 | 377.30 | EXIT_EMA400 | -27.65 |
| SELL | 2025-04-24 13:15:00 | 326.65 | 2025-04-30 13:15:00 | 307.92 | TARGET | 18.73 |
| SELL | 2025-05-08 12:15:00 | 322.95 | 2025-05-09 09:15:00 | 306.69 | TARGET | 16.26 |
| SELL | 2025-04-15 11:15:00 | 296.50 | 2025-05-09 11:15:00 | 330.50 | EXIT_EMA400 | -34.00 |
| BUY | 2025-09-08 10:15:00 | 513.75 | 2025-10-23 15:15:00 | 627.42 | TARGET | 113.67 |
| BUY | 2025-12-09 11:15:00 | 621.80 | 2025-12-18 09:15:00 | 603.00 | EXIT_EMA400 | -18.80 |
| SELL | 2026-02-04 11:15:00 | 503.45 | 2026-02-12 09:15:00 | 536.35 | EXIT_EMA400 | -32.90 |
