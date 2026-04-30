# IIFL Finance Ltd. (IIFL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 460.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 1
- **Winners / losers:** 0 / 5
- **Target hits / EMA400 exits:** 0 / 5
- **Total realized P&L (per unit):** -138.60
- **Avg P&L per closed trade:** -27.72

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 14:15:00 | 401.50 | 446.75 | 446.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 15:15:00 | 400.25 | 446.29 | 446.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-19 14:15:00 | 452.05 | 442.92 | 444.94 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 14:15:00 | 468.85 | 446.83 | 446.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 09:15:00 | 469.65 | 448.17 | 447.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 10:15:00 | 449.15 | 450.32 | 448.69 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-30 10:15:00 | 457.95 | 450.29 | 448.73 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 449.90 | 450.72 | 449.00 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-09-02 11:15:00 | 446.85 | 450.68 | 448.99 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 09:15:00 | 422.50 | 464.10 | 464.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 09:15:00 | 393.35 | 460.91 | 462.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 447.65 | 445.58 | 453.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-12 12:15:00 | 436.45 | 446.52 | 452.51 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-04 09:15:00 | 447.10 | 430.30 | 440.13 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 378.75 | 348.46 | 348.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 387.70 | 351.46 | 350.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 14:15:00 | 498.55 | 500.73 | 469.34 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-29 14:15:00 | 513.60 | 500.84 | 470.48 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-31 15:15:00 | 471.75 | 500.20 | 472.36 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 11:15:00 | 437.90 | 459.83 | 459.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 435.55 | 458.34 | 459.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 10:15:00 | 451.75 | 450.88 | 454.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-16 12:15:00 | 446.50 | 450.83 | 454.59 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-18 11:15:00 | 454.30 | 450.32 | 454.09 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 12:15:00 | 495.45 | 454.74 | 454.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 504.20 | 463.17 | 459.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-21 10:15:00 | 611.00 | 613.45 | 582.61 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-21 12:15:00 | 624.90 | 613.55 | 582.96 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 593.30 | 613.40 | 583.64 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-22 14:15:00 | 557.70 | 612.19 | 583.62 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 503.60 | 564.07 | 564.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 10:15:00 | 502.85 | 562.92 | 563.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 13:15:00 | 469.20 | 465.09 | 489.57 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-23 09:15:00 | 447.85 | 466.72 | 486.24 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-08-30 10:15:00 | 457.95 | 2024-09-02 11:15:00 | 446.85 | EXIT_EMA400 | -11.10 |
| SELL | 2024-11-12 12:15:00 | 436.45 | 2024-12-04 09:15:00 | 447.10 | EXIT_EMA400 | -10.65 |
| BUY | 2025-07-29 14:15:00 | 513.60 | 2025-07-31 15:15:00 | 471.75 | EXIT_EMA400 | -41.85 |
| SELL | 2025-09-16 12:15:00 | 446.50 | 2025-09-18 11:15:00 | 454.30 | EXIT_EMA400 | -7.80 |
| BUY | 2026-01-21 12:15:00 | 624.90 | 2026-01-22 14:15:00 | 557.70 | EXIT_EMA400 | -67.20 |
