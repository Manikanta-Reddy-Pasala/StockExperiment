# Firstsource Solutions Ltd. (FSL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 213.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 1 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 2 / 2
- **Total realized P&L (per unit):** -0.48
- **Avg P&L per closed trade:** -0.12

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 14:15:00 | 335.00 | 358.19 | 358.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 10:15:00 | 330.00 | 354.59 | 355.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 10:15:00 | 355.40 | 354.29 | 355.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-10 09:15:00 | 346.40 | 354.54 | 355.68 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-27 14:15:00 | 359.70 | 330.44 | 340.34 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 374.90 | 339.54 | 339.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 10:15:00 | 378.30 | 342.84 | 341.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 373.80 | 374.32 | 363.76 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 09:15:00 | 384.80 | 374.39 | 364.69 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-01 11:15:00 | 361.05 | 376.83 | 367.72 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 13:15:00 | 346.10 | 363.68 | 363.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 345.70 | 362.87 | 363.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 10:15:00 | 353.00 | 351.98 | 356.99 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 15:15:00 | 369.00 | 359.86 | 359.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 09:15:00 | 375.95 | 360.02 | 359.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 13:15:00 | 358.00 | 360.82 | 360.32 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 10:15:00 | 348.25 | 359.77 | 359.81 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 12:15:00 | 368.00 | 359.88 | 359.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 14:15:00 | 369.05 | 360.04 | 359.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 10:15:00 | 355.00 | 360.29 | 360.06 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 09:15:00 | 349.65 | 359.79 | 359.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 10:15:00 | 347.30 | 359.67 | 359.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 362.85 | 358.08 | 358.90 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 12:15:00 | 363.90 | 359.63 | 359.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 13:15:00 | 364.55 | 359.68 | 359.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 10:15:00 | 362.25 | 362.55 | 361.21 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 14:15:00 | 327.25 | 359.88 | 360.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-01 10:15:00 | 323.80 | 356.85 | 358.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 343.00 | 336.17 | 344.81 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 12:15:00 | 358.85 | 349.02 | 349.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 13:15:00 | 359.95 | 349.13 | 349.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 350.15 | 351.05 | 350.07 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-21 12:15:00 | 359.60 | 351.19 | 350.15 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 350.85 | 351.52 | 350.36 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-24 14:15:00 | 341.05 | 351.42 | 350.31 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 339.85 | 349.62 | 349.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 14:15:00 | 338.60 | 349.51 | 349.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 346.80 | 345.51 | 347.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-26 14:15:00 | 343.15 | 347.02 | 347.81 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-16 09:15:00 | 251.10 | 226.03 | 247.29 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-10 09:15:00 | 346.40 | 2025-03-12 10:15:00 | 318.56 | TARGET | 27.84 |
| BUY | 2025-06-24 09:15:00 | 384.80 | 2025-07-01 11:15:00 | 361.05 | EXIT_EMA400 | -23.75 |
| BUY | 2025-11-21 12:15:00 | 359.60 | 2025-11-24 14:15:00 | 341.05 | EXIT_EMA400 | -18.55 |
| SELL | 2025-12-26 14:15:00 | 343.15 | 2025-12-30 14:15:00 | 329.17 | TARGET | 13.98 |
