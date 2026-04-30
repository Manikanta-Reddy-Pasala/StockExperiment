# International Gemmological Institute (India) Ltd. (IGIL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-12-20 09:15:00 → 2026-04-30 15:30:00 (2330 bars)
- **Last close:** 345.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT3 | 4 |
| ENTRY1 | 2 |
| ENTRY2 | 4 |
| EXIT | 2 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -31.74
- **Avg P&L per closed trade:** -5.29

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 11:15:00 | 397.60 | 379.76 | 379.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 09:15:00 | 405.15 | 383.62 | 381.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 385.70 | 391.90 | 386.88 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 14:15:00 | 344.00 | 383.02 | 383.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 09:15:00 | 333.90 | 382.14 | 382.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 374.40 | 354.33 | 363.43 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-26 09:15:00 | 350.90 | 363.45 | 366.00 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 357.60 | 359.82 | 363.71 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-03 11:15:00 | 356.00 | 359.78 | 363.65 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 348.15 | 356.35 | 361.07 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-13 11:15:00 | 347.20 | 356.26 | 361.00 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 350.85 | 350.05 | 356.69 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-24 09:15:00 | 340.55 | 349.86 | 356.37 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-11-06 09:15:00 | 357.40 | 345.35 | 352.19 | Close above EMA400 |

### Cycle 3 — BUY (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 15:15:00 | 335.95 | 325.65 | 325.62 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 312.80 | 325.65 | 325.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 14:15:00 | 304.85 | 325.32 | 325.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 14:15:00 | 323.55 | 323.27 | 324.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-17 15:15:00 | 318.95 | 323.23 | 324.39 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 318.95 | 323.23 | 324.39 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-03-18 09:15:00 | 315.10 | 323.15 | 324.34 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-03-18 13:15:00 | 327.60 | 323.16 | 324.33 | Close above EMA400 |

### Cycle 5 — BUY (started 2026-04-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 10:15:00 | 340.15 | 325.06 | 325.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 346.85 | 326.75 | 325.91 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-10-03 11:15:00 | 356.00 | 2025-10-15 10:15:00 | 333.04 | TARGET | 22.96 |
| SELL | 2025-09-26 09:15:00 | 350.90 | 2025-11-06 09:15:00 | 357.40 | EXIT_EMA400 | -6.50 |
| SELL | 2025-10-13 11:15:00 | 347.20 | 2025-11-06 09:15:00 | 357.40 | EXIT_EMA400 | -10.20 |
| SELL | 2025-10-24 09:15:00 | 340.55 | 2025-11-06 09:15:00 | 357.40 | EXIT_EMA400 | -16.85 |
| SELL | 2026-03-17 15:15:00 | 318.95 | 2026-03-18 13:15:00 | 327.60 | EXIT_EMA400 | -8.65 |
| SELL | 2026-03-18 09:15:00 | 315.10 | 2026-03-18 13:15:00 | 327.60 | EXIT_EMA400 | -12.50 |
