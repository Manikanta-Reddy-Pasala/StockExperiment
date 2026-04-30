# Bharat Electronics Ltd. (BEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 431.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 4 |
| ENTRY1 | 7 |
| ENTRY2 | 4 |
| EXIT | 7 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 2 / 9
- **Target hits / EMA400 exits:** 2 / 9
- **Total realized P&L (per unit):** -9.17
- **Avg P&L per closed trade:** -0.83

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 15:15:00 | 284.40 | 295.39 | 295.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 09:15:00 | 283.75 | 295.28 | 295.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-24 13:15:00 | 292.15 | 291.30 | 293.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-25 09:15:00 | 289.50 | 291.29 | 293.13 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 289.50 | 291.29 | 293.13 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-25 11:15:00 | 289.30 | 291.26 | 293.10 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 291.85 | 291.14 | 292.93 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-27 10:15:00 | 291.15 | 291.14 | 292.92 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-09-27 14:15:00 | 293.75 | 291.16 | 292.89 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 09:15:00 | 308.60 | 287.77 | 287.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 11:15:00 | 309.95 | 291.73 | 289.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 11:15:00 | 303.05 | 303.44 | 297.44 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-18 15:15:00 | 304.50 | 303.44 | 297.55 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 297.65 | 303.38 | 297.55 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-12-19 11:15:00 | 300.40 | 303.30 | 297.57 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 298.25 | 303.17 | 297.59 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-12-20 10:15:00 | 300.65 | 303.05 | 297.61 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-12-20 12:15:00 | 296.35 | 302.96 | 297.62 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 14:15:00 | 281.30 | 294.66 | 294.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 273.85 | 294.32 | 294.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 12:15:00 | 286.90 | 286.88 | 290.26 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-20 15:15:00 | 285.55 | 286.85 | 290.20 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-31 09:15:00 | 288.20 | 279.51 | 285.21 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 15:15:00 | 303.10 | 276.72 | 276.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-26 11:15:00 | 304.61 | 279.04 | 277.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 09:15:00 | 275.50 | 283.61 | 280.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-15 09:15:00 | 293.00 | 282.63 | 280.59 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-28 12:15:00 | 384.40 | 402.56 | 386.65 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 09:15:00 | 375.20 | 382.29 | 382.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 372.70 | 381.44 | 381.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 10:15:00 | 381.05 | 379.46 | 380.75 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 397.65 | 381.90 | 381.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 400.75 | 382.69 | 382.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 14:15:00 | 407.00 | 407.56 | 400.13 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-30 15:15:00 | 410.20 | 407.67 | 400.48 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-24 10:15:00 | 408.05 | 416.02 | 408.64 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 11:15:00 | 390.90 | 405.68 | 405.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 13:15:00 | 389.90 | 405.37 | 405.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 400.50 | 400.36 | 402.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-29 14:15:00 | 393.55 | 400.31 | 402.46 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-02 09:15:00 | 404.85 | 399.50 | 401.79 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 10:15:00 | 420.05 | 403.77 | 403.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 430.40 | 408.91 | 406.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 440.90 | 444.04 | 433.21 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-16 10:15:00 | 454.30 | 433.49 | 431.03 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-04-27 11:15:00 | 435.20 | 440.16 | 435.42 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-25 09:15:00 | 289.50 | 2024-09-27 14:15:00 | 293.75 | EXIT_EMA400 | -4.25 |
| SELL | 2024-09-25 11:15:00 | 289.30 | 2024-09-27 14:15:00 | 293.75 | EXIT_EMA400 | -4.45 |
| SELL | 2024-09-27 10:15:00 | 291.15 | 2024-09-27 14:15:00 | 293.75 | EXIT_EMA400 | -2.60 |
| BUY | 2024-12-18 15:15:00 | 304.50 | 2024-12-20 12:15:00 | 296.35 | EXIT_EMA400 | -8.15 |
| BUY | 2024-12-19 11:15:00 | 300.40 | 2024-12-20 12:15:00 | 296.35 | EXIT_EMA400 | -4.05 |
| BUY | 2024-12-20 10:15:00 | 300.65 | 2024-12-20 12:15:00 | 296.35 | EXIT_EMA400 | -4.30 |
| SELL | 2025-01-20 15:15:00 | 285.55 | 2025-01-22 09:15:00 | 271.61 | TARGET | 13.94 |
| BUY | 2025-04-15 09:15:00 | 293.00 | 2025-05-13 09:15:00 | 330.24 | TARGET | 37.24 |
| BUY | 2025-10-30 15:15:00 | 410.20 | 2025-11-24 10:15:00 | 408.05 | EXIT_EMA400 | -2.15 |
| SELL | 2025-12-29 14:15:00 | 393.55 | 2026-01-02 09:15:00 | 404.85 | EXIT_EMA400 | -11.30 |
| BUY | 2026-04-16 10:15:00 | 454.30 | 2026-04-27 11:15:00 | 435.20 | EXIT_EMA400 | -19.10 |
