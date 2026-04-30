# Honasa Consumer Ltd. (HONASA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 340.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT3 | 6 |
| ENTRY1 | 8 |
| ENTRY2 | 1 |
| EXIT | 8 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 0 / 9
- **Target hits / EMA400 exits:** 0 / 9
- **Total realized P&L (per unit):** -112.46
- **Avg P&L per closed trade:** -12.50

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 15:15:00 | 429.95 | 472.34 | 472.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-08 09:15:00 | 421.20 | 471.83 | 472.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 14:15:00 | 233.73 | 230.96 | 257.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-14 12:15:00 | 219.10 | 230.59 | 256.25 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 12:15:00 | 234.44 | 220.52 | 234.77 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-24 13:15:00 | 235.36 | 220.67 | 234.78 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 250.18 | 236.51 | 236.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 10:15:00 | 255.46 | 238.65 | 237.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 297.45 | 297.65 | 279.68 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 11:15:00 | 307.20 | 298.22 | 280.94 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-14 10:15:00 | 287.55 | 299.72 | 288.77 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 13:15:00 | 264.90 | 283.81 | 283.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 10:15:00 | 264.10 | 283.10 | 283.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 294.45 | 276.78 | 279.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-14 09:15:00 | 279.80 | 277.56 | 280.21 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 279.80 | 277.56 | 280.21 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-08-14 10:15:00 | 284.45 | 277.63 | 280.23 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 299.55 | 282.28 | 282.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 13:15:00 | 302.90 | 283.03 | 282.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 11:15:00 | 297.00 | 297.02 | 292.22 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-25 13:15:00 | 299.40 | 297.04 | 292.28 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-26 09:15:00 | 284.85 | 296.89 | 292.28 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 276.80 | 289.41 | 289.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 09:15:00 | 276.50 | 289.08 | 289.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 09:15:00 | 283.75 | 283.40 | 286.05 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-06 14:15:00 | 276.55 | 282.89 | 285.47 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 278.10 | 281.08 | 284.23 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-13 09:15:00 | 293.30 | 281.23 | 284.19 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 10:15:00 | 293.05 | 286.26 | 286.25 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 10:15:00 | 279.80 | 286.24 | 286.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 12:15:00 | 279.30 | 286.11 | 286.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 14:15:00 | 272.00 | 271.78 | 277.31 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-26 13:15:00 | 269.75 | 271.95 | 277.05 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 275.80 | 271.90 | 276.82 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-30 09:15:00 | 283.90 | 272.06 | 276.86 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 13:15:00 | 296.00 | 280.42 | 280.39 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 15:15:00 | 271.50 | 280.87 | 280.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 267.35 | 280.39 | 280.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 13:15:00 | 280.55 | 277.70 | 279.14 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-02 10:15:00 | 270.55 | 277.62 | 279.07 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 275.55 | 277.42 | 278.93 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-04 13:15:00 | 279.40 | 277.33 | 278.80 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 11:15:00 | 292.65 | 280.01 | 279.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 09:15:00 | 295.50 | 280.59 | 280.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 10:15:00 | 291.10 | 293.84 | 288.38 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-05 09:15:00 | 294.50 | 293.41 | 288.50 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 294.50 | 293.41 | 288.50 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-05 14:15:00 | 299.50 | 293.44 | 288.64 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-06 15:15:00 | 288.20 | 293.59 | 288.90 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 10:15:00 | 280.80 | 285.77 | 285.79 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 14:15:00 | 289.50 | 285.82 | 285.81 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 281.55 | 285.76 | 285.78 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 291.25 | 285.81 | 285.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 293.30 | 285.88 | 285.83 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-14 12:15:00 | 219.10 | 2025-03-24 13:15:00 | 235.36 | EXIT_EMA400 | -16.26 |
| BUY | 2025-06-24 11:15:00 | 307.20 | 2025-07-14 10:15:00 | 287.55 | EXIT_EMA400 | -19.65 |
| SELL | 2025-08-14 09:15:00 | 279.80 | 2025-08-14 10:15:00 | 284.45 | EXIT_EMA400 | -4.65 |
| BUY | 2025-09-25 13:15:00 | 299.40 | 2025-09-26 09:15:00 | 284.85 | EXIT_EMA400 | -14.55 |
| SELL | 2025-11-06 14:15:00 | 276.55 | 2025-11-13 09:15:00 | 293.30 | EXIT_EMA400 | -16.75 |
| SELL | 2025-12-26 13:15:00 | 269.75 | 2025-12-30 09:15:00 | 283.90 | EXIT_EMA400 | -14.15 |
| SELL | 2026-02-02 10:15:00 | 270.55 | 2026-02-04 13:15:00 | 279.40 | EXIT_EMA400 | -8.85 |
| BUY | 2026-03-05 09:15:00 | 294.50 | 2026-03-06 15:15:00 | 288.20 | EXIT_EMA400 | -6.30 |
| BUY | 2026-03-05 14:15:00 | 299.50 | 2026-03-06 15:15:00 | 288.20 | EXIT_EMA400 | -11.30 |
