# Power Grid Corporation of India Ltd. (POWERGRID.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 319.00
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
| ENTRY2 | 3 |
| EXIT | 6 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 4 / 5
- **Target hits / EMA400 exits:** 3 / 6
- **Total realized P&L (per unit):** -10.08
- **Avg P&L per closed trade:** -1.12

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 15:15:00 | 331.05 | 336.41 | 336.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 329.65 | 336.35 | 336.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 329.75 | 324.98 | 329.45 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-12 13:15:00 | 322.90 | 325.33 | 329.41 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-22 09:15:00 | 330.60 | 322.89 | 327.36 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 09:15:00 | 306.00 | 285.86 | 285.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 13:15:00 | 306.80 | 286.65 | 286.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 299.85 | 300.83 | 295.30 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 308.15 | 300.85 | 295.50 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 297.10 | 301.16 | 296.08 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-14 12:15:00 | 294.95 | 301.10 | 296.08 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 288.10 | 294.84 | 294.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 10:15:00 | 287.05 | 294.51 | 294.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 14:15:00 | 293.95 | 292.68 | 293.62 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 299.15 | 294.37 | 294.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 12:15:00 | 299.50 | 294.46 | 294.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 13:15:00 | 294.15 | 295.80 | 295.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-21 11:15:00 | 296.30 | 295.76 | 295.18 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 296.30 | 295.76 | 295.18 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-21 12:15:00 | 297.00 | 295.77 | 295.19 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 296.50 | 295.81 | 295.22 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-22 13:15:00 | 298.30 | 295.86 | 295.26 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-25 09:15:00 | 294.35 | 296.26 | 295.51 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 290.40 | 294.92 | 294.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 287.15 | 294.84 | 294.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 13:15:00 | 291.15 | 290.96 | 292.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-18 14:15:00 | 290.30 | 290.96 | 292.57 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 288.30 | 285.75 | 288.51 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-12 15:15:00 | 288.60 | 285.86 | 288.43 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 294.40 | 288.17 | 288.16 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 278.50 | 288.17 | 288.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 271.35 | 287.82 | 288.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 11:15:00 | 268.65 | 268.51 | 274.00 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-23 14:15:00 | 267.10 | 268.49 | 273.91 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 269.10 | 267.18 | 272.07 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-06 11:15:00 | 267.40 | 267.61 | 271.96 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-02 13:15:00 | 266.90 | 260.37 | 265.29 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 11:15:00 | 292.80 | 269.48 | 269.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 13:15:00 | 293.80 | 269.95 | 269.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 289.75 | 290.18 | 282.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-09 11:15:00 | 293.25 | 290.22 | 282.89 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-04-02 09:15:00 | 286.60 | 295.14 | 289.21 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-12 13:15:00 | 322.90 | 2024-11-22 09:15:00 | 330.60 | EXIT_EMA400 | -7.70 |
| BUY | 2025-05-12 09:15:00 | 308.15 | 2025-05-14 12:15:00 | 294.95 | EXIT_EMA400 | -13.20 |
| BUY | 2025-07-21 11:15:00 | 296.30 | 2025-07-24 09:15:00 | 299.66 | TARGET | 3.36 |
| BUY | 2025-07-21 12:15:00 | 297.00 | 2025-07-25 09:15:00 | 294.35 | EXIT_EMA400 | -2.65 |
| BUY | 2025-07-22 13:15:00 | 298.30 | 2025-07-25 09:15:00 | 294.35 | EXIT_EMA400 | -3.95 |
| SELL | 2025-08-18 14:15:00 | 290.30 | 2025-08-22 09:15:00 | 283.48 | TARGET | 6.82 |
| SELL | 2026-01-06 11:15:00 | 267.40 | 2026-01-20 14:15:00 | 253.72 | TARGET | 13.68 |
| SELL | 2025-12-23 14:15:00 | 267.10 | 2026-02-02 13:15:00 | 266.90 | EXIT_EMA400 | 0.20 |
| BUY | 2026-03-09 11:15:00 | 293.25 | 2026-04-02 09:15:00 | 286.60 | EXIT_EMA400 | -6.65 |
