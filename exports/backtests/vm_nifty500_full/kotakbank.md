# Kotak Mahindra Bank Ltd. (KOTAKBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 383.30
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 7 |
| ENTRY1 | 7 |
| ENTRY2 | 4 |
| EXIT | 7 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 1 / 10
- **Target hits / EMA400 exits:** 1 / 10
- **Total realized P&L (per unit):** -43.20
- **Avg P&L per closed trade:** -3.93

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 15:15:00 | 367.69 | 354.35 | 354.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 09:15:00 | 372.00 | 354.53 | 354.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 09:15:00 | 367.74 | 368.65 | 363.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-09 10:15:00 | 371.34 | 368.50 | 363.72 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 365.78 | 368.47 | 363.85 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-01-11 09:15:00 | 366.72 | 368.21 | 363.87 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 13:15:00 | 363.97 | 368.12 | 363.91 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-01-12 12:15:00 | 366.89 | 367.95 | 363.95 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-01-17 09:15:00 | 362.52 | 368.15 | 364.40 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 10:15:00 | 347.48 | 362.19 | 362.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 13:15:00 | 345.21 | 361.74 | 361.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 09:15:00 | 350.55 | 349.45 | 354.04 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2024-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 12:15:00 | 360.07 | 353.98 | 353.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 09:15:00 | 360.89 | 354.22 | 354.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 09:15:00 | 332.76 | 356.34 | 355.25 | EMA200 retest candle locked |

### Cycle 4 — SELL (started 2024-04-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 11:15:00 | 324.16 | 353.90 | 354.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 13:15:00 | 324.01 | 353.33 | 353.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-17 10:15:00 | 338.00 | 337.07 | 343.39 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-31 09:15:00 | 334.14 | 338.45 | 342.51 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 341.53 | 338.29 | 342.29 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-06-03 10:15:00 | 343.22 | 338.34 | 342.29 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 355.07 | 344.06 | 344.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 12:15:00 | 357.62 | 344.42 | 344.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 348.64 | 349.11 | 346.79 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-03 09:15:00 | 357.90 | 349.32 | 346.95 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 354.88 | 358.66 | 353.45 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-07-22 10:15:00 | 353.15 | 358.60 | 353.45 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 13:15:00 | 348.29 | 362.65 | 362.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 14:15:00 | 346.74 | 362.50 | 362.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 13:15:00 | 353.00 | 352.61 | 356.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-03 09:15:00 | 349.00 | 353.16 | 356.03 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-05 12:15:00 | 355.88 | 352.87 | 355.64 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 12:15:00 | 384.92 | 356.16 | 356.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 14:15:00 | 385.35 | 369.09 | 363.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 10:15:00 | 383.08 | 383.34 | 375.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-03-05 10:15:00 | 387.44 | 383.17 | 375.81 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 413.00 | 427.84 | 412.75 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-12 09:15:00 | 431.30 | 425.19 | 413.71 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-05-15 09:15:00 | 414.20 | 424.74 | 414.63 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 390.78 | 424.39 | 424.40 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 11:15:00 | 428.30 | 407.84 | 407.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 12:15:00 | 429.18 | 408.06 | 407.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 09:15:00 | 423.78 | 424.06 | 417.90 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-01 09:15:00 | 431.24 | 420.59 | 418.57 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 429.80 | 432.01 | 427.47 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-07 09:15:00 | 426.96 | 431.90 | 427.48 | Close below EMA400 |

### Cycle 10 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 410.40 | 425.23 | 425.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 10:15:00 | 407.00 | 424.13 | 424.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 11:15:00 | 422.70 | 420.38 | 422.57 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-19 14:15:00 | 416.15 | 422.54 | 423.27 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 421.65 | 422.40 | 423.17 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-20 15:15:00 | 420.15 | 422.37 | 423.15 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-23 09:15:00 | 427.45 | 422.42 | 423.18 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-09 10:15:00 | 371.34 | 2024-01-17 09:15:00 | 362.52 | EXIT_EMA400 | -8.82 |
| BUY | 2024-01-11 09:15:00 | 366.72 | 2024-01-17 09:15:00 | 362.52 | EXIT_EMA400 | -4.20 |
| BUY | 2024-01-12 12:15:00 | 366.89 | 2024-01-17 09:15:00 | 362.52 | EXIT_EMA400 | -4.37 |
| SELL | 2024-05-31 09:15:00 | 334.14 | 2024-06-03 10:15:00 | 343.22 | EXIT_EMA400 | -9.08 |
| BUY | 2024-07-03 09:15:00 | 357.90 | 2024-07-22 10:15:00 | 353.15 | EXIT_EMA400 | -4.75 |
| SELL | 2024-12-03 09:15:00 | 349.00 | 2024-12-05 12:15:00 | 355.88 | EXIT_EMA400 | -6.88 |
| BUY | 2025-03-05 10:15:00 | 387.44 | 2025-03-24 09:15:00 | 422.32 | TARGET | 34.88 |
| BUY | 2025-05-12 09:15:00 | 431.30 | 2025-05-15 09:15:00 | 414.20 | EXIT_EMA400 | -17.10 |
| BUY | 2025-12-01 09:15:00 | 431.24 | 2026-01-07 09:15:00 | 426.96 | EXIT_EMA400 | -4.28 |
| SELL | 2026-02-19 14:15:00 | 416.15 | 2026-02-23 09:15:00 | 427.45 | EXIT_EMA400 | -11.30 |
| SELL | 2026-02-20 15:15:00 | 420.15 | 2026-02-23 09:15:00 | 427.45 | EXIT_EMA400 | -7.30 |
