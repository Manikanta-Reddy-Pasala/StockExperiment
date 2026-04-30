# International Gemmological Institute (India) Ltd. (IGIL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-12-20 09:15:00 → 2026-04-30 15:15:00 (2347 bars)
- **Last close:** 344.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 6 |
| ENTRY1 | 5 |
| ENTRY2 | 5 |
| EXIT | 4 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 1
- **Winners / losers:** 1 / 8
- **Target hits / EMA400 exits:** 1 / 8
- **Total realized P&L (per unit):** -55.44
- **Avg P&L per closed trade:** -6.16

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 11:15:00 | 397.80 | 379.74 | 379.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 09:15:00 | 404.45 | 383.63 | 381.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 385.70 | 391.86 | 386.85 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 344.00 | 382.62 | 382.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 09:15:00 | 334.15 | 382.13 | 382.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 374.40 | 354.35 | 363.43 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-26 09:15:00 | 350.90 | 363.40 | 365.97 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 357.60 | 359.78 | 363.68 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-03 11:15:00 | 356.00 | 359.74 | 363.62 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 348.15 | 356.32 | 361.04 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-13 11:15:00 | 347.20 | 356.23 | 360.98 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 350.75 | 350.01 | 356.66 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-24 09:15:00 | 340.55 | 349.83 | 356.34 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-11-06 09:15:00 | 357.90 | 345.35 | 352.18 | Close above EMA400 |

### Cycle 3 — BUY (started 2026-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 12:15:00 | 329.40 | 325.32 | 325.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 13:15:00 | 330.00 | 325.37 | 325.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 325.00 | 325.47 | 325.39 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-02 12:15:00 | 326.50 | 325.47 | 325.40 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 12:15:00 | 326.50 | 325.47 | 325.40 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-02 14:15:00 | 325.25 | 325.48 | 325.40 | Close below EMA400 |

### Cycle 4 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 317.15 | 325.29 | 325.31 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 329.15 | 325.35 | 325.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 13:15:00 | 330.90 | 325.44 | 325.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 320.50 | 326.52 | 325.96 | EMA200 retest candle locked |

### Cycle 6 — SELL (started 2026-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 14:15:00 | 303.70 | 325.31 | 325.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 299.20 | 324.89 | 325.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 14:15:00 | 324.10 | 323.28 | 324.31 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-17 15:15:00 | 318.95 | 323.24 | 324.28 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 318.95 | 323.24 | 324.28 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-03-18 09:15:00 | 315.10 | 323.15 | 324.24 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-03-18 13:15:00 | 327.60 | 323.17 | 324.22 | Close above EMA400 |

### Cycle 7 — BUY (started 2026-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 14:15:00 | 333.25 | 325.12 | 325.12 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 321.30 | 325.08 | 325.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 12:15:00 | 320.30 | 325.03 | 325.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 327.65 | 324.35 | 324.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-02 09:15:00 | 315.60 | 324.33 | 324.70 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 320.20 | 324.04 | 324.54 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-06 09:15:00 | 317.45 | 323.94 | 324.48 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-08 09:15:00 | 326.95 | 323.44 | 324.19 | Close above EMA400 |

### Cycle 9 — BUY (started 2026-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 09:15:00 | 339.40 | 324.89 | 324.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 346.85 | 326.74 | 325.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 10:15:00 | 342.80 | 343.11 | 336.03 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-30 13:15:00 | 345.95 | 343.15 | 336.15 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-10-03 11:15:00 | 356.00 | 2025-10-15 10:15:00 | 333.14 | TARGET | 22.86 |
| SELL | 2025-09-26 09:15:00 | 350.90 | 2025-11-06 09:15:00 | 357.90 | EXIT_EMA400 | -7.00 |
| SELL | 2025-10-13 11:15:00 | 347.20 | 2025-11-06 09:15:00 | 357.90 | EXIT_EMA400 | -10.70 |
| SELL | 2025-10-24 09:15:00 | 340.55 | 2025-11-06 09:15:00 | 357.90 | EXIT_EMA400 | -17.35 |
| BUY | 2026-03-02 12:15:00 | 326.50 | 2026-03-02 14:15:00 | 325.25 | EXIT_EMA400 | -1.25 |
| SELL | 2026-03-17 15:15:00 | 318.95 | 2026-03-18 13:15:00 | 327.60 | EXIT_EMA400 | -8.65 |
| SELL | 2026-03-18 09:15:00 | 315.10 | 2026-03-18 13:15:00 | 327.60 | EXIT_EMA400 | -12.50 |
| SELL | 2026-04-02 09:15:00 | 315.60 | 2026-04-08 09:15:00 | 326.95 | EXIT_EMA400 | -11.35 |
| SELL | 2026-04-06 09:15:00 | 317.45 | 2026-04-08 09:15:00 | 326.95 | EXIT_EMA400 | -9.50 |
