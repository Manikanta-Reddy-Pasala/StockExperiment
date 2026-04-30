# Bharat Electronics Ltd. (BEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 431.30
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 5 |
| ENTRY1 | 7 |
| ENTRY2 | 5 |
| EXIT | 7 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 2 / 10
- **Target hits / EMA400 exits:** 2 / 10
- **Total realized P&L (per unit):** -15.68
- **Avg P&L per closed trade:** -1.31

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 15:15:00 | 282.65 | 294.51 | 294.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 274.95 | 294.32 | 294.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-24 13:15:00 | 292.15 | 291.29 | 292.79 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-25 09:15:00 | 289.50 | 291.29 | 292.76 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 289.50 | 291.29 | 292.76 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-25 11:15:00 | 289.30 | 291.25 | 292.73 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 291.85 | 291.14 | 292.58 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-27 10:15:00 | 291.15 | 291.14 | 292.58 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-09-27 14:15:00 | 293.75 | 291.15 | 292.56 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 09:15:00 | 308.60 | 287.76 | 287.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 11:15:00 | 310.00 | 291.72 | 289.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 11:15:00 | 303.10 | 303.45 | 297.40 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-18 15:15:00 | 304.50 | 303.45 | 297.52 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 297.65 | 303.39 | 297.52 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-12-19 11:15:00 | 300.40 | 303.31 | 297.54 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 298.25 | 303.18 | 297.55 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-12-20 10:15:00 | 300.65 | 303.06 | 297.58 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-12-20 12:15:00 | 296.35 | 302.97 | 297.59 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 15:15:00 | 281.25 | 294.53 | 294.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 273.85 | 294.33 | 294.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 12:15:00 | 286.90 | 286.86 | 290.23 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-20 15:15:00 | 285.45 | 286.83 | 290.17 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-31 09:15:00 | 288.25 | 279.50 | 285.19 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 15:15:00 | 303.10 | 276.69 | 276.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-26 11:15:00 | 304.59 | 279.01 | 277.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 09:15:00 | 275.40 | 283.57 | 280.45 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-15 09:15:00 | 293.00 | 282.61 | 280.58 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-28 12:15:00 | 384.40 | 402.55 | 386.64 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 09:15:00 | 375.20 | 382.29 | 382.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 372.70 | 381.43 | 381.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 10:15:00 | 381.05 | 379.46 | 380.74 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 397.65 | 381.89 | 381.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 400.75 | 382.68 | 382.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 14:15:00 | 406.95 | 407.55 | 400.13 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-30 15:15:00 | 410.05 | 407.66 | 400.47 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-24 10:15:00 | 408.05 | 416.00 | 408.63 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 11:15:00 | 390.95 | 405.67 | 405.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 13:15:00 | 389.85 | 405.37 | 405.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 400.50 | 400.36 | 402.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-29 14:15:00 | 393.60 | 400.32 | 402.46 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-02 09:15:00 | 404.85 | 399.51 | 401.80 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 10:15:00 | 420.00 | 403.78 | 403.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 430.30 | 408.90 | 406.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 440.90 | 443.83 | 432.86 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-16 10:15:00 | 454.30 | 433.42 | 430.85 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 435.35 | 439.41 | 434.82 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-04-28 09:15:00 | 439.95 | 439.32 | 434.86 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-04-28 13:15:00 | 434.10 | 439.18 | 434.89 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-25 09:15:00 | 289.50 | 2024-09-27 14:15:00 | 293.75 | EXIT_EMA400 | -4.25 |
| SELL | 2024-09-25 11:15:00 | 289.30 | 2024-09-27 14:15:00 | 293.75 | EXIT_EMA400 | -4.45 |
| SELL | 2024-09-27 10:15:00 | 291.15 | 2024-09-27 14:15:00 | 293.75 | EXIT_EMA400 | -2.60 |
| BUY | 2024-12-18 15:15:00 | 304.50 | 2024-12-20 12:15:00 | 296.35 | EXIT_EMA400 | -8.15 |
| BUY | 2024-12-19 11:15:00 | 300.40 | 2024-12-20 12:15:00 | 296.35 | EXIT_EMA400 | -4.05 |
| BUY | 2024-12-20 10:15:00 | 300.65 | 2024-12-20 12:15:00 | 296.35 | EXIT_EMA400 | -4.30 |
| SELL | 2025-01-20 15:15:00 | 285.45 | 2025-01-22 09:15:00 | 271.30 | TARGET | 14.15 |
| BUY | 2025-04-15 09:15:00 | 293.00 | 2025-05-13 09:15:00 | 330.27 | TARGET | 37.27 |
| BUY | 2025-10-30 15:15:00 | 410.05 | 2025-11-24 10:15:00 | 408.05 | EXIT_EMA400 | -2.00 |
| SELL | 2025-12-29 14:15:00 | 393.60 | 2026-01-02 09:15:00 | 404.85 | EXIT_EMA400 | -11.25 |
| BUY | 2026-04-16 10:15:00 | 454.30 | 2026-04-28 13:15:00 | 434.10 | EXIT_EMA400 | -20.20 |
| BUY | 2026-04-28 09:15:00 | 439.95 | 2026-04-28 13:15:00 | 434.10 | EXIT_EMA400 | -5.85 |
