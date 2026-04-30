# EIH Ltd. (EIHOTEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 319.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 1 / 7
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -30.93
- **Avg P&L per closed trade:** -3.87

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 12:15:00 | 433.90 | 383.35 | 383.31 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 13:15:00 | 353.05 | 396.84 | 396.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 09:15:00 | 337.50 | 382.88 | 388.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-06 09:15:00 | 345.85 | 345.54 | 362.79 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-06 11:15:00 | 339.25 | 345.46 | 362.58 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-13 14:15:00 | 359.50 | 344.59 | 359.11 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 12:15:00 | 387.70 | 364.35 | 364.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 13:15:00 | 389.30 | 364.60 | 364.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 14:15:00 | 368.00 | 369.26 | 367.05 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-16 12:15:00 | 371.15 | 366.11 | 365.83 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 369.45 | 369.35 | 367.76 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-27 15:15:00 | 370.00 | 369.36 | 367.77 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-05-28 11:15:00 | 367.55 | 369.35 | 367.79 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 347.60 | 368.07 | 368.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 345.25 | 367.84 | 367.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 13:15:00 | 365.30 | 363.30 | 365.48 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-27 12:15:00 | 360.95 | 363.39 | 365.39 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 360.95 | 363.39 | 365.39 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-06-27 14:15:00 | 357.65 | 363.32 | 365.33 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-06-30 09:15:00 | 365.40 | 363.29 | 365.30 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 15:15:00 | 382.20 | 366.72 | 366.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 390.15 | 369.65 | 368.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 13:15:00 | 373.85 | 374.57 | 371.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-29 09:15:00 | 378.60 | 374.64 | 371.45 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 377.85 | 375.24 | 371.98 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-31 11:15:00 | 380.40 | 375.32 | 372.05 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 372.95 | 375.64 | 372.41 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-04 10:15:00 | 372.10 | 375.61 | 372.41 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 14:15:00 | 377.30 | 386.51 | 386.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 09:15:00 | 375.35 | 386.31 | 386.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 380.20 | 379.68 | 382.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-15 11:15:00 | 374.25 | 379.32 | 381.71 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-05 12:15:00 | 365.10 | 338.23 | 351.85 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-06 11:15:00 | 339.25 | 2025-03-13 14:15:00 | 359.50 | EXIT_EMA400 | -20.25 |
| BUY | 2025-05-16 12:15:00 | 371.15 | 2025-05-28 11:15:00 | 367.55 | EXIT_EMA400 | -3.60 |
| BUY | 2025-05-27 15:15:00 | 370.00 | 2025-05-28 11:15:00 | 367.55 | EXIT_EMA400 | -2.45 |
| SELL | 2025-06-27 12:15:00 | 360.95 | 2025-06-30 09:15:00 | 365.40 | EXIT_EMA400 | -4.45 |
| SELL | 2025-06-27 14:15:00 | 357.65 | 2025-06-30 09:15:00 | 365.40 | EXIT_EMA400 | -7.75 |
| BUY | 2025-07-29 09:15:00 | 378.60 | 2025-08-04 10:15:00 | 372.10 | EXIT_EMA400 | -6.50 |
| BUY | 2025-07-31 11:15:00 | 380.40 | 2025-08-04 10:15:00 | 372.10 | EXIT_EMA400 | -8.30 |
| SELL | 2025-12-15 11:15:00 | 374.25 | 2026-01-08 10:15:00 | 351.88 | TARGET | 22.37 |
