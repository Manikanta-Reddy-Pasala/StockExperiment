# Mahindra & Mahindra Financial Services Ltd. (M&MFIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 311.45
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 4 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 1
- **Winners / losers:** 4 / 3
- **Target hits / EMA400 exits:** 4 / 3
- **Total realized P&L (per unit):** 26.46
- **Avg P&L per closed trade:** 3.78

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 14:15:00 | 290.20 | 307.70 | 307.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 284.55 | 307.29 | 307.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 10:15:00 | 278.00 | 275.21 | 285.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-25 12:15:00 | 272.40 | 275.18 | 285.71 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-03 09:15:00 | 286.95 | 274.62 | 283.52 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 09:15:00 | 299.35 | 277.02 | 276.92 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 11:15:00 | 271.20 | 277.70 | 277.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 12:15:00 | 269.70 | 277.45 | 277.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 11:15:00 | 276.15 | 275.93 | 276.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-10 14:15:00 | 272.40 | 276.10 | 276.78 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 273.95 | 274.82 | 276.03 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-18 09:15:00 | 278.70 | 274.87 | 276.01 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 10:15:00 | 288.30 | 277.04 | 277.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 13:15:00 | 291.45 | 277.41 | 277.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 275.80 | 280.93 | 279.20 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 15:15:00 | 262.30 | 277.73 | 277.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 245.50 | 277.41 | 277.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-16 09:15:00 | 275.00 | 272.39 | 274.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-23 09:15:00 | 267.25 | 272.96 | 274.76 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 267.25 | 272.96 | 274.76 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-23 10:15:00 | 266.60 | 272.89 | 274.72 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 267.10 | 264.40 | 268.60 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-05-16 15:15:00 | 266.30 | 264.42 | 268.59 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 268.00 | 264.53 | 268.57 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-05-19 13:15:00 | 266.90 | 264.55 | 268.56 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 263.85 | 262.10 | 265.85 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-06-06 10:15:00 | 274.10 | 262.38 | 265.61 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 13:15:00 | 272.05 | 262.83 | 262.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 276.50 | 263.98 | 263.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 11:15:00 | 272.35 | 273.03 | 268.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-30 10:15:00 | 275.80 | 273.08 | 269.00 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-09 13:15:00 | 348.00 | 368.03 | 348.53 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 11:15:00 | 317.80 | 360.01 | 360.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 314.50 | 355.56 | 357.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 314.75 | 309.36 | 324.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-30 14:15:00 | 309.55 | 311.03 | 323.65 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-25 12:15:00 | 272.40 | 2024-12-03 09:15:00 | 286.95 | EXIT_EMA400 | -14.55 |
| SELL | 2025-03-10 14:15:00 | 272.40 | 2025-03-18 09:15:00 | 278.70 | EXIT_EMA400 | -6.30 |
| SELL | 2025-04-23 09:15:00 | 267.25 | 2025-05-09 11:15:00 | 244.71 | TARGET | 22.54 |
| SELL | 2025-05-19 13:15:00 | 266.90 | 2025-05-20 09:15:00 | 261.92 | TARGET | 4.98 |
| SELL | 2025-05-16 15:15:00 | 266.30 | 2025-05-20 11:15:00 | 259.42 | TARGET | 6.88 |
| SELL | 2025-04-23 10:15:00 | 266.60 | 2025-06-06 10:15:00 | 274.10 | EXIT_EMA400 | -7.50 |
| BUY | 2025-09-30 10:15:00 | 275.80 | 2025-10-16 09:15:00 | 296.21 | TARGET | 20.41 |
