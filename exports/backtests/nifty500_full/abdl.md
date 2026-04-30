# Allied Blenders and Distillers Ltd. (ABDL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-07-02 09:15:00 → 2026-04-30 15:15:00 (3148 bars)
- **Last close:** 531.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -18.37
- **Avg P&L per closed trade:** -2.62

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 10:15:00 | 303.60 | 329.32 | 329.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 13:15:00 | 299.30 | 328.54 | 329.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 14:15:00 | 331.75 | 326.99 | 328.15 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-30 13:15:00 | 321.00 | 327.03 | 328.14 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 327.70 | 327.04 | 328.13 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-04 15:15:00 | 320.50 | 326.66 | 327.86 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-11-05 13:15:00 | 328.35 | 326.56 | 327.78 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 14:15:00 | 351.95 | 327.10 | 327.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 12:15:00 | 364.20 | 329.98 | 328.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 11:15:00 | 396.00 | 400.65 | 376.74 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-15 10:15:00 | 404.95 | 399.75 | 377.78 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-27 09:15:00 | 377.30 | 400.56 | 383.60 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 11:15:00 | 325.65 | 377.59 | 377.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 316.10 | 362.56 | 369.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 09:15:00 | 326.10 | 321.40 | 335.55 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-15 11:15:00 | 296.50 | 321.23 | 335.32 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 329.05 | 320.88 | 332.93 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-23 14:15:00 | 333.05 | 321.39 | 332.89 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 09:15:00 | 401.00 | 335.21 | 335.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 15:15:00 | 433.00 | 379.85 | 363.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 498.35 | 498.69 | 473.43 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-08 10:15:00 | 513.75 | 498.48 | 475.82 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 603.20 | 627.05 | 599.98 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-09 11:15:00 | 621.80 | 626.83 | 600.14 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-12-18 09:15:00 | 603.00 | 624.35 | 604.37 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 507.60 | 597.31 | 597.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 10:15:00 | 504.20 | 596.39 | 597.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 514.75 | 510.86 | 542.15 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-05 09:15:00 | 502.40 | 510.58 | 540.93 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-12 09:15:00 | 536.35 | 510.80 | 536.17 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 12:15:00 | 552.50 | 490.62 | 490.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 14:15:00 | 558.15 | 491.88 | 491.03 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-30 13:15:00 | 321.00 | 2024-11-05 13:15:00 | 328.35 | EXIT_EMA400 | -7.35 |
| SELL | 2024-11-04 15:15:00 | 320.50 | 2024-11-05 13:15:00 | 328.35 | EXIT_EMA400 | -7.85 |
| BUY | 2025-01-15 10:15:00 | 404.95 | 2025-01-27 09:15:00 | 377.30 | EXIT_EMA400 | -27.65 |
| SELL | 2025-04-15 11:15:00 | 296.50 | 2025-04-23 14:15:00 | 333.05 | EXIT_EMA400 | -36.55 |
| BUY | 2025-09-08 10:15:00 | 513.75 | 2025-10-23 15:15:00 | 627.53 | TARGET | 113.78 |
| BUY | 2025-12-09 11:15:00 | 621.80 | 2025-12-18 09:15:00 | 603.00 | EXIT_EMA400 | -18.80 |
| SELL | 2026-02-05 09:15:00 | 502.40 | 2026-02-12 09:15:00 | 536.35 | EXIT_EMA400 | -33.95 |
