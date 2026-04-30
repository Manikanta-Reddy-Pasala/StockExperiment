# Power Grid Corporation of India Ltd. (POWERGRID.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 318.35
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 5 |
| ENTRY1 | 6 |
| ENTRY2 | 4 |
| EXIT | 6 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 4 / 6
- **Target hits / EMA400 exits:** 3 / 7
- **Total realized P&L (per unit):** -12.80
- **Avg P&L per closed trade:** -1.28

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 09:15:00 | 318.60 | 335.51 | 335.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 09:15:00 | 313.85 | 332.12 | 333.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 329.75 | 325.08 | 329.29 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-12 13:15:00 | 322.90 | 325.42 | 329.26 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-22 09:15:00 | 330.60 | 322.95 | 327.23 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 10:15:00 | 305.00 | 286.08 | 286.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 13:15:00 | 306.80 | 286.67 | 286.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 299.90 | 300.84 | 295.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 308.10 | 300.86 | 295.56 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 297.10 | 301.17 | 296.13 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-14 12:15:00 | 294.90 | 301.11 | 296.13 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 288.15 | 294.84 | 294.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 10:15:00 | 287.05 | 294.51 | 294.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 14:15:00 | 293.90 | 292.69 | 293.64 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 299.10 | 294.42 | 294.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 09:15:00 | 301.15 | 294.68 | 294.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 13:15:00 | 294.15 | 295.82 | 295.21 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-21 11:15:00 | 296.30 | 295.77 | 295.20 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 296.30 | 295.77 | 295.20 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-21 12:15:00 | 297.00 | 295.78 | 295.21 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 296.50 | 295.82 | 295.24 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-22 13:15:00 | 298.30 | 295.87 | 295.28 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-25 09:15:00 | 294.35 | 296.28 | 295.54 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 290.40 | 294.92 | 294.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 287.15 | 294.84 | 294.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 13:15:00 | 291.15 | 290.96 | 292.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-18 14:15:00 | 290.30 | 290.95 | 292.58 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 288.30 | 285.75 | 288.51 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-12 12:15:00 | 285.70 | 285.80 | 288.44 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-12 15:15:00 | 288.60 | 285.86 | 288.43 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 294.40 | 288.16 | 288.15 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 278.50 | 288.16 | 288.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 271.35 | 287.81 | 288.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 11:15:00 | 268.70 | 268.51 | 274.00 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-23 14:15:00 | 267.10 | 268.49 | 273.91 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 269.10 | 267.19 | 272.08 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-06 11:15:00 | 267.40 | 267.63 | 271.97 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-02 13:15:00 | 266.75 | 260.79 | 265.67 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 11:15:00 | 292.80 | 269.76 | 269.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 13:15:00 | 293.85 | 270.22 | 269.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 289.80 | 290.27 | 282.98 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-09 11:15:00 | 293.25 | 290.31 | 283.07 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-04-02 09:15:00 | 286.65 | 295.19 | 289.33 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-12 13:15:00 | 322.90 | 2024-11-22 09:15:00 | 330.60 | EXIT_EMA400 | -7.70 |
| BUY | 2025-05-12 09:15:00 | 308.10 | 2025-05-14 12:15:00 | 294.90 | EXIT_EMA400 | -13.20 |
| BUY | 2025-07-21 11:15:00 | 296.30 | 2025-07-23 10:15:00 | 299.60 | TARGET | 3.30 |
| BUY | 2025-07-21 12:15:00 | 297.00 | 2025-07-25 09:15:00 | 294.35 | EXIT_EMA400 | -2.65 |
| BUY | 2025-07-22 13:15:00 | 298.30 | 2025-07-25 09:15:00 | 294.35 | EXIT_EMA400 | -3.95 |
| SELL | 2025-08-18 14:15:00 | 290.30 | 2025-08-22 09:15:00 | 283.47 | TARGET | 6.83 |
| SELL | 2025-09-12 12:15:00 | 285.70 | 2025-09-12 15:15:00 | 288.60 | EXIT_EMA400 | -2.90 |
| SELL | 2026-01-06 11:15:00 | 267.40 | 2026-01-20 14:15:00 | 253.69 | TARGET | 13.71 |
| SELL | 2025-12-23 14:15:00 | 267.10 | 2026-02-02 13:15:00 | 266.75 | EXIT_EMA400 | 0.35 |
| BUY | 2026-03-09 11:15:00 | 293.25 | 2026-04-02 09:15:00 | 286.65 | EXIT_EMA400 | -6.60 |
