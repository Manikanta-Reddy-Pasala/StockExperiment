# Nuvoco Vistas Corporation Ltd. (NUVOCO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 296.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 17 |
| ALERT1 | 14 |
| ALERT2 | 14 |
| ALERT3 | 10 |
| ENTRY1 | 12 |
| ENTRY2 | 3 |
| EXIT | 11 |

## P&L

- **Trades closed:** 15
- **Trades open at end:** 0
- **Winners / losers:** 4 / 11
- **Target hits / EMA400 exits:** 4 / 11
- **Total realized P&L (per unit):** 24.94
- **Avg P&L per closed trade:** 1.66

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 10:15:00 | 344.35 | 351.02 | 351.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-23 13:15:00 | 341.80 | 350.79 | 350.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-04 09:15:00 | 350.35 | 346.42 | 348.41 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-09-04 11:15:00 | 347.30 | 346.47 | 348.41 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 11:15:00 | 347.30 | 346.47 | 348.41 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-09-04 12:15:00 | 350.55 | 346.51 | 348.42 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 15:15:00 | 370.90 | 350.18 | 350.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-07 09:15:00 | 378.55 | 350.47 | 350.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-05 09:15:00 | 367.00 | 367.28 | 361.26 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2023-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 15:15:00 | 344.70 | 359.45 | 359.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 10:15:00 | 343.00 | 359.14 | 359.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-16 13:15:00 | 354.60 | 350.59 | 354.06 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2023-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 15:15:00 | 370.10 | 355.93 | 355.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 09:15:00 | 374.65 | 356.12 | 355.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 14:15:00 | 365.00 | 365.41 | 361.61 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-21 14:15:00 | 376.80 | 365.60 | 361.84 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-01-03 13:15:00 | 364.80 | 371.28 | 366.01 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-01-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 14:15:00 | 342.40 | 362.85 | 362.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 340.50 | 362.42 | 362.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-29 09:15:00 | 355.75 | 355.57 | 358.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-09 09:15:00 | 350.65 | 356.83 | 358.72 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 15:15:00 | 327.75 | 319.92 | 331.00 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-04-04 09:15:00 | 324.80 | 319.97 | 330.97 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 14:15:00 | 324.65 | 317.51 | 326.41 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-04-19 15:15:00 | 326.60 | 317.60 | 326.41 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 11:15:00 | 350.00 | 326.57 | 326.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 09:15:00 | 363.70 | 327.80 | 327.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 349.75 | 351.49 | 343.75 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-29 09:15:00 | 358.30 | 348.78 | 343.85 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 345.05 | 349.49 | 344.72 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-08-02 09:15:00 | 340.05 | 349.26 | 344.76 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 10:15:00 | 330.95 | 341.39 | 341.44 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 14:15:00 | 350.15 | 341.16 | 341.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 10:15:00 | 354.35 | 342.33 | 341.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 09:15:00 | 350.75 | 352.97 | 348.94 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-08 10:15:00 | 355.20 | 352.92 | 349.07 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 353.05 | 352.90 | 349.12 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-09 09:15:00 | 358.70 | 352.96 | 349.21 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 351.00 | 353.63 | 350.08 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-15 12:15:00 | 349.80 | 353.57 | 350.09 | Close below EMA400 |

### Cycle 9 — SELL (started 2024-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 15:15:00 | 342.55 | 348.22 | 348.24 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 13:15:00 | 352.20 | 348.26 | 348.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-04 11:15:00 | 357.25 | 348.55 | 348.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 11:15:00 | 349.15 | 349.97 | 349.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-11-08 14:15:00 | 352.00 | 349.96 | 349.20 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 352.00 | 349.96 | 349.20 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-11-11 09:15:00 | 347.00 | 349.94 | 349.19 | Close below EMA400 |

### Cycle 11 — SELL (started 2024-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 12:15:00 | 332.00 | 348.48 | 348.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 328.20 | 348.12 | 348.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 14:15:00 | 346.25 | 344.35 | 346.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-26 14:15:00 | 339.85 | 344.18 | 346.02 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-28 09:15:00 | 346.80 | 344.08 | 345.89 | Close above EMA400 |

### Cycle 12 — BUY (started 2024-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 10:15:00 | 367.95 | 347.40 | 347.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 10:15:00 | 369.45 | 348.72 | 348.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 11:15:00 | 356.50 | 356.91 | 353.07 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-20 10:15:00 | 362.40 | 356.91 | 353.31 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-12-20 13:15:00 | 352.70 | 356.90 | 353.36 | Close below EMA400 |

### Cycle 13 — SELL (started 2025-01-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 13:15:00 | 350.00 | 352.07 | 352.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 13:15:00 | 340.45 | 351.83 | 351.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 352.90 | 351.84 | 351.96 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-28 09:15:00 | 335.60 | 350.92 | 351.46 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 346.50 | 349.59 | 350.74 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-31 14:15:00 | 351.00 | 349.39 | 350.55 | Close above EMA400 |

### Cycle 14 — BUY (started 2025-02-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 15:15:00 | 358.60 | 351.52 | 351.50 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 12:15:00 | 350.25 | 351.47 | 351.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 13:15:00 | 347.85 | 351.43 | 351.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 09:15:00 | 328.55 | 315.22 | 326.81 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 298.85 | 315.18 | 323.47 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-15 09:15:00 | 322.60 | 313.52 | 321.47 | Close above EMA400 |

### Cycle 16 — BUY (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 10:15:00 | 344.15 | 325.68 | 325.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 345.20 | 326.72 | 326.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 13:15:00 | 327.40 | 327.84 | 326.73 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 342.50 | 328.02 | 326.84 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 344.30 | 350.51 | 344.10 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-19 11:15:00 | 341.05 | 350.42 | 344.08 | Close below EMA400 |

### Cycle 17 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 397.85 | 423.69 | 423.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 391.05 | 422.86 | 423.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 10:15:00 | 363.90 | 363.75 | 382.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-15 11:15:00 | 358.75 | 363.70 | 382.00 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 358.00 | 349.98 | 358.43 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-12 10:15:00 | 350.85 | 350.15 | 358.35 | Sell entry 2 (retest2 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-09-04 11:15:00 | 347.30 | 2023-09-04 12:15:00 | 350.55 | EXIT_EMA400 | -3.25 |
| BUY | 2023-12-21 14:15:00 | 376.80 | 2024-01-03 13:15:00 | 364.80 | EXIT_EMA400 | -12.00 |
| SELL | 2024-02-09 09:15:00 | 350.65 | 2024-03-06 09:15:00 | 326.43 | TARGET | 24.22 |
| SELL | 2024-04-04 09:15:00 | 324.80 | 2024-04-15 09:15:00 | 306.28 | TARGET | 18.52 |
| BUY | 2024-07-29 09:15:00 | 358.30 | 2024-08-02 09:15:00 | 340.05 | EXIT_EMA400 | -18.25 |
| BUY | 2024-10-08 10:15:00 | 355.20 | 2024-10-15 12:15:00 | 349.80 | EXIT_EMA400 | -5.40 |
| BUY | 2024-10-09 09:15:00 | 358.70 | 2024-10-15 12:15:00 | 349.80 | EXIT_EMA400 | -8.90 |
| BUY | 2024-11-08 14:15:00 | 352.00 | 2024-11-11 09:15:00 | 347.00 | EXIT_EMA400 | -5.00 |
| SELL | 2024-11-26 14:15:00 | 339.85 | 2024-11-28 09:15:00 | 346.80 | EXIT_EMA400 | -6.95 |
| BUY | 2024-12-20 10:15:00 | 362.40 | 2024-12-20 13:15:00 | 352.70 | EXIT_EMA400 | -9.70 |
| SELL | 2025-01-28 09:15:00 | 335.60 | 2025-01-31 14:15:00 | 351.00 | EXIT_EMA400 | -15.40 |
| SELL | 2025-04-07 09:15:00 | 298.85 | 2025-04-15 09:15:00 | 322.60 | EXIT_EMA400 | -23.75 |
| BUY | 2025-05-12 09:15:00 | 342.50 | 2025-06-19 11:15:00 | 341.05 | EXIT_EMA400 | -1.45 |
| SELL | 2026-02-12 10:15:00 | 350.85 | 2026-02-27 09:15:00 | 328.36 | TARGET | 22.49 |
| SELL | 2025-12-15 11:15:00 | 358.75 | 2026-03-09 09:15:00 | 288.99 | TARGET | 69.76 |
