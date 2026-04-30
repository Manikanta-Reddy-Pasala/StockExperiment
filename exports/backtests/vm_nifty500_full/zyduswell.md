# Zydus Wellness Ltd. (ZYDUSWELL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 508.55
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT3 | 8 |
| ENTRY1 | 8 |
| ENTRY2 | 1 |
| EXIT | 8 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 3 / 6
- **Target hits / EMA400 exits:** 3 / 6
- **Total realized P&L (per unit):** -10.83
- **Avg P&L per closed trade:** -1.20

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 15:15:00 | 304.61 | 310.98 | 310.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-07 11:15:00 | 302.65 | 310.68 | 310.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-17 09:15:00 | 309.16 | 308.98 | 309.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-12-13 09:15:00 | 304.63 | 309.89 | 310.08 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 307.46 | 309.64 | 309.95 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-12-14 13:15:00 | 313.41 | 309.61 | 309.93 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-18 10:15:00 | 322.34 | 310.25 | 310.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 10:15:00 | 324.21 | 313.11 | 311.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 12:15:00 | 323.26 | 324.99 | 319.99 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-30 13:15:00 | 327.60 | 323.60 | 320.32 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 320.40 | 323.57 | 320.32 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-02-01 14:15:00 | 320.22 | 323.42 | 320.47 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-02-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 15:15:00 | 313.48 | 319.34 | 319.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-01 10:15:00 | 312.21 | 319.22 | 319.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-05 09:15:00 | 318.94 | 318.68 | 319.00 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-12 15:15:00 | 312.00 | 318.14 | 318.66 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 309.57 | 306.29 | 311.19 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-04-04 11:15:00 | 312.84 | 306.64 | 311.15 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 12:15:00 | 332.28 | 314.05 | 314.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 14:15:00 | 334.55 | 317.09 | 315.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 337.00 | 340.38 | 332.24 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-04 14:15:00 | 346.42 | 340.41 | 332.42 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 435.96 | 448.22 | 435.08 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-09-17 14:15:00 | 433.97 | 447.83 | 435.08 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 15:15:00 | 390.00 | 426.73 | 426.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 13:15:00 | 387.98 | 424.95 | 425.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 10:15:00 | 391.18 | 391.02 | 403.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-31 11:15:00 | 387.37 | 390.98 | 403.56 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 399.91 | 391.14 | 401.78 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-08 09:15:00 | 397.43 | 391.21 | 401.75 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 395.23 | 388.23 | 396.53 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-27 12:15:00 | 399.34 | 388.34 | 396.55 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 13:15:00 | 359.82 | 347.33 | 347.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 362.60 | 347.72 | 347.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 13:15:00 | 405.34 | 406.22 | 394.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-30 13:15:00 | 409.24 | 405.92 | 395.12 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 398.14 | 406.68 | 396.84 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-05 14:15:00 | 396.52 | 406.39 | 396.84 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 14:15:00 | 430.25 | 454.55 | 454.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 10:15:00 | 429.30 | 453.83 | 454.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 15:15:00 | 434.45 | 433.72 | 441.14 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-23 10:15:00 | 423.85 | 433.57 | 440.99 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-31 09:15:00 | 460.10 | 431.64 | 438.78 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 14:15:00 | 475.30 | 444.75 | 444.62 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 13:15:00 | 437.30 | 444.61 | 444.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 432.95 | 444.36 | 444.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 10:15:00 | 437.00 | 435.67 | 439.51 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-02 09:15:00 | 431.25 | 436.05 | 439.59 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 431.25 | 436.05 | 439.59 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-02 15:15:00 | 439.95 | 435.85 | 439.38 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 10:15:00 | 503.10 | 418.25 | 418.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 11:15:00 | 522.90 | 419.29 | 418.77 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-12-13 09:15:00 | 304.63 | 2023-12-14 13:15:00 | 313.41 | EXIT_EMA400 | -8.78 |
| BUY | 2024-01-30 13:15:00 | 327.60 | 2024-02-01 14:15:00 | 320.22 | EXIT_EMA400 | -7.38 |
| SELL | 2024-03-12 15:15:00 | 312.00 | 2024-03-14 14:15:00 | 292.02 | TARGET | 19.98 |
| BUY | 2024-06-04 14:15:00 | 346.42 | 2024-06-05 10:15:00 | 388.43 | TARGET | 42.01 |
| SELL | 2024-11-08 09:15:00 | 397.43 | 2024-11-11 09:15:00 | 384.46 | TARGET | 12.97 |
| SELL | 2024-10-31 11:15:00 | 387.37 | 2024-11-27 12:15:00 | 399.34 | EXIT_EMA400 | -11.97 |
| BUY | 2025-07-30 13:15:00 | 409.24 | 2025-08-05 14:15:00 | 396.52 | EXIT_EMA400 | -12.72 |
| SELL | 2025-12-23 10:15:00 | 423.85 | 2025-12-31 09:15:00 | 460.10 | EXIT_EMA400 | -36.25 |
| SELL | 2026-02-02 09:15:00 | 431.25 | 2026-02-02 15:15:00 | 439.95 | EXIT_EMA400 | -8.70 |
