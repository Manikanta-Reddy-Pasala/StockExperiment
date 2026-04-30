# Rail Vikas Nigam Ltd. (RVNL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 298.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** -28.40
- **Avg P&L per closed trade:** -7.10

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 10:15:00 | 476.50 | 522.42 | 522.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 11:15:00 | 473.60 | 521.93 | 522.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 09:15:00 | 478.65 | 475.10 | 492.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 09:15:00 | 455.00 | 475.16 | 491.47 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-06 14:15:00 | 465.65 | 447.74 | 465.11 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 11:15:00 | 409.85 | 369.16 | 369.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 14:15:00 | 416.15 | 379.44 | 374.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 400.40 | 403.74 | 391.23 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 10:15:00 | 375.10 | 389.71 | 389.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 12:15:00 | 374.50 | 389.41 | 389.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 12:15:00 | 338.35 | 338.27 | 355.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-04 12:15:00 | 326.50 | 337.35 | 353.50 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-15 09:15:00 | 352.80 | 336.12 | 349.45 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 362.30 | 329.99 | 329.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 11:15:00 | 365.45 | 335.22 | 332.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 10:15:00 | 340.75 | 341.17 | 336.30 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-01 09:15:00 | 348.85 | 334.94 | 334.21 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 344.00 | 335.03 | 334.26 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-02-01 12:15:00 | 329.95 | 334.99 | 334.24 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-03 09:15:00 | 323.70 | 333.50 | 333.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 10:15:00 | 321.35 | 332.77 | 333.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 321.60 | 320.02 | 325.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-25 13:15:00 | 315.75 | 319.92 | 324.90 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 288.30 | 276.67 | 290.42 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-16 10:15:00 | 296.60 | 276.87 | 290.45 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-08 09:15:00 | 455.00 | 2024-12-06 14:15:00 | 465.65 | EXIT_EMA400 | -10.65 |
| SELL | 2025-09-04 12:15:00 | 326.50 | 2025-09-15 09:15:00 | 352.80 | EXIT_EMA400 | -26.30 |
| BUY | 2026-02-01 09:15:00 | 348.85 | 2026-02-01 12:15:00 | 329.95 | EXIT_EMA400 | -18.90 |
| SELL | 2026-02-25 13:15:00 | 315.75 | 2026-03-04 09:15:00 | 288.30 | TARGET | 27.45 |
