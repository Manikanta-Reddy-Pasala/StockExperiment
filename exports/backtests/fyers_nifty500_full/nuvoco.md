# Nuvoco Vistas Corporation Ltd. (NUVOCO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 295.05
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -25.80
- **Avg P&L per closed trade:** -6.45

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 15:15:00 | 328.05 | 348.11 | 348.13 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 09:15:00 | 367.00 | 347.24 | 347.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 12:15:00 | 371.80 | 350.57 | 348.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 11:15:00 | 356.50 | 356.97 | 353.04 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-20 10:15:00 | 362.40 | 356.98 | 353.29 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 356.05 | 357.02 | 353.35 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-20 13:15:00 | 352.70 | 356.98 | 353.34 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 351.00 | 352.06 | 352.06 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 11:15:00 | 353.85 | 352.08 | 352.07 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 12:15:00 | 349.55 | 352.05 | 352.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 13:15:00 | 340.45 | 351.93 | 352.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 352.95 | 351.94 | 352.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-28 09:15:00 | 335.60 | 350.99 | 351.50 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 346.50 | 349.66 | 350.78 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-31 14:15:00 | 351.00 | 349.45 | 350.60 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 09:15:00 | 341.55 | 325.56 | 325.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 345.20 | 326.79 | 326.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 13:15:00 | 327.40 | 327.87 | 326.75 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 342.50 | 328.07 | 326.86 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 344.30 | 350.49 | 344.10 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-19 11:15:00 | 341.05 | 350.40 | 344.08 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 14:15:00 | 398.50 | 423.47 | 423.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 391.05 | 422.89 | 423.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 10:15:00 | 363.90 | 363.74 | 382.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-15 11:15:00 | 358.75 | 363.69 | 382.00 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-11 13:15:00 | 358.00 | 349.32 | 357.74 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-12-20 10:15:00 | 362.40 | 2024-12-20 13:15:00 | 352.70 | EXIT_EMA400 | -9.70 |
| SELL | 2025-01-28 09:15:00 | 335.60 | 2025-01-31 14:15:00 | 351.00 | EXIT_EMA400 | -15.40 |
| BUY | 2025-05-12 09:15:00 | 342.50 | 2025-06-19 11:15:00 | 341.05 | EXIT_EMA400 | -1.45 |
| SELL | 2025-12-15 11:15:00 | 358.75 | 2026-02-11 13:15:00 | 358.00 | EXIT_EMA400 | 0.75 |
