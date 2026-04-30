# Firstsource Solutions Ltd. (FSL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 213.94
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 12 |
| ALERT2 | 12 |
| ALERT3 | 1 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** -11.66
- **Avg P&L per closed trade:** -2.33

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 193.45 | 197.55 | 197.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 13:15:00 | 193.15 | 197.50 | 197.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-07 09:15:00 | 199.00 | 193.82 | 195.47 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 12:15:00 | 202.60 | 196.69 | 196.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 09:15:00 | 210.34 | 196.98 | 196.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 09:15:00 | 303.05 | 304.19 | 281.71 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-20 10:15:00 | 312.80 | 304.94 | 285.15 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-12-23 11:15:00 | 346.55 | 363.48 | 349.36 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 14:15:00 | 352.40 | 358.31 | 358.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 13:15:00 | 349.80 | 358.07 | 358.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 14:15:00 | 359.15 | 357.26 | 357.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-12 09:15:00 | 345.00 | 357.12 | 357.70 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 345.00 | 357.12 | 357.70 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-02-12 11:15:00 | 362.30 | 357.11 | 357.69 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 14:15:00 | 373.35 | 339.91 | 339.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 10:15:00 | 378.30 | 342.89 | 341.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 373.80 | 374.32 | 363.79 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 09:15:00 | 384.80 | 374.39 | 364.73 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-01 11:15:00 | 361.05 | 376.80 | 367.73 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 13:15:00 | 346.10 | 363.69 | 363.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 345.70 | 362.88 | 363.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 10:15:00 | 353.00 | 351.99 | 357.00 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 14:15:00 | 368.00 | 359.82 | 359.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 09:15:00 | 375.95 | 360.06 | 359.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 13:15:00 | 358.35 | 360.87 | 360.36 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 10:15:00 | 348.25 | 359.80 | 359.84 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 12:15:00 | 368.00 | 359.90 | 359.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 14:15:00 | 369.05 | 360.06 | 359.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 10:15:00 | 355.10 | 360.31 | 360.08 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 09:15:00 | 349.65 | 359.81 | 359.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 10:15:00 | 347.30 | 359.68 | 359.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 362.85 | 358.09 | 358.91 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 13:15:00 | 364.45 | 359.68 | 359.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 14:15:00 | 365.70 | 359.74 | 359.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 10:15:00 | 362.25 | 362.55 | 361.22 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2025-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 14:15:00 | 327.25 | 359.89 | 360.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-01 10:15:00 | 323.95 | 356.85 | 358.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 343.00 | 336.19 | 344.83 | EMA200 retest candle locked |

### Cycle 12 — BUY (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 12:15:00 | 358.85 | 349.02 | 349.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 13:15:00 | 359.95 | 349.13 | 349.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 350.15 | 351.06 | 350.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-21 12:15:00 | 359.60 | 351.20 | 350.16 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-24 14:15:00 | 341.25 | 351.44 | 350.33 | Close below EMA400 |

### Cycle 13 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 340.10 | 349.60 | 349.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 14:15:00 | 338.60 | 349.49 | 349.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 346.80 | 345.52 | 347.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-26 14:15:00 | 343.15 | 347.02 | 347.81 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-16 09:15:00 | 251.10 | 226.06 | 247.42 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-09-20 10:15:00 | 312.80 | 2024-12-23 11:15:00 | 346.55 | EXIT_EMA400 | 33.75 |
| SELL | 2025-02-12 09:15:00 | 345.00 | 2025-02-12 11:15:00 | 362.30 | EXIT_EMA400 | -17.30 |
| BUY | 2025-06-24 09:15:00 | 384.80 | 2025-07-01 11:15:00 | 361.05 | EXIT_EMA400 | -23.75 |
| BUY | 2025-11-21 12:15:00 | 359.60 | 2025-11-24 14:15:00 | 341.25 | EXIT_EMA400 | -18.35 |
| SELL | 2025-12-26 14:15:00 | 343.15 | 2025-12-30 14:15:00 | 329.16 | TARGET | 13.99 |
