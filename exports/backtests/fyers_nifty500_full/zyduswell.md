# Zydus Wellness Ltd. (ZYDUSWELL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 506.15
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** -40.84
- **Avg P&L per closed trade:** -8.17

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 13:15:00 | 385.19 | 427.51 | 427.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 383.90 | 421.57 | 424.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 10:15:00 | 391.18 | 391.04 | 403.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-31 11:15:00 | 387.37 | 391.00 | 403.83 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-07 15:15:00 | 401.98 | 391.17 | 401.90 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 12:15:00 | 359.22 | 347.19 | 347.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 362.60 | 347.73 | 347.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 13:15:00 | 405.34 | 406.24 | 394.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-30 13:15:00 | 409.24 | 405.92 | 395.11 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 398.14 | 406.63 | 396.80 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-05 14:15:00 | 396.52 | 406.34 | 396.81 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 14:15:00 | 430.25 | 454.54 | 454.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 10:15:00 | 429.30 | 453.82 | 454.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 15:15:00 | 434.45 | 433.76 | 441.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-23 10:15:00 | 423.85 | 433.61 | 441.02 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-31 09:15:00 | 460.10 | 431.65 | 438.79 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 14:15:00 | 475.30 | 444.75 | 444.63 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 13:15:00 | 437.30 | 444.62 | 444.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 432.95 | 444.37 | 444.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 10:15:00 | 437.00 | 435.68 | 439.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-02 09:15:00 | 431.25 | 436.49 | 439.70 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 431.25 | 436.49 | 439.70 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-03 12:15:00 | 419.05 | 435.85 | 439.21 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-03-13 11:15:00 | 421.65 | 397.39 | 410.81 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 522.80 | 419.18 | 418.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 09:15:00 | 531.55 | 439.94 | 430.02 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-31 11:15:00 | 387.37 | 2024-11-07 15:15:00 | 401.98 | EXIT_EMA400 | -14.61 |
| BUY | 2025-07-30 13:15:00 | 409.24 | 2025-08-05 14:15:00 | 396.52 | EXIT_EMA400 | -12.72 |
| SELL | 2025-12-23 10:15:00 | 423.85 | 2025-12-31 09:15:00 | 460.10 | EXIT_EMA400 | -36.25 |
| SELL | 2026-02-02 09:15:00 | 431.25 | 2026-02-04 10:15:00 | 405.91 | TARGET | 25.34 |
| SELL | 2026-02-03 12:15:00 | 419.05 | 2026-03-13 11:15:00 | 421.65 | EXIT_EMA400 | -2.60 |
