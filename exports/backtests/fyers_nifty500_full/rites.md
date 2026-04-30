# RITES Ltd. (RITES.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 218.30
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 4 |
| ENTRY1 | 3 |
| ENTRY2 | 2 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / EMA400 exits:** 3 / 2
- **Total realized P&L (per unit):** 23.02
- **Avg P&L per closed trade:** 4.60

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 11:15:00 | 331.05 | 347.24 | 347.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 09:15:00 | 328.93 | 346.42 | 346.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 339.15 | 336.51 | 340.91 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 11:15:00 | 364.55 | 343.05 | 342.95 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-10-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 15:15:00 | 316.75 | 343.49 | 343.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 306.15 | 343.12 | 343.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 301.40 | 291.98 | 306.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-26 11:15:00 | 285.80 | 291.96 | 306.20 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-09 09:15:00 | 304.35 | 290.70 | 301.71 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 11:15:00 | 296.34 | 234.98 | 234.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 305.35 | 262.47 | 251.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 276.45 | 278.14 | 264.22 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 10:15:00 | 280.40 | 277.30 | 265.16 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 273.15 | 278.93 | 272.80 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-23 11:15:00 | 272.70 | 278.81 | 272.80 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 250.40 | 268.93 | 269.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 248.70 | 268.55 | 268.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 14:15:00 | 258.29 | 257.39 | 261.69 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 15:15:00 | 270.11 | 263.92 | 263.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 11:15:00 | 271.50 | 264.11 | 264.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 263.10 | 264.56 | 264.24 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 10:15:00 | 255.91 | 263.88 | 263.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 11:15:00 | 254.76 | 263.79 | 263.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 11:15:00 | 249.71 | 249.39 | 253.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-13 11:15:00 | 246.47 | 249.33 | 253.44 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 252.83 | 248.99 | 253.01 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-18 09:15:00 | 248.78 | 249.15 | 252.97 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 252.50 | 249.23 | 252.63 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-21 14:15:00 | 247.29 | 249.24 | 252.56 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 238.31 | 233.18 | 240.32 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-23 11:15:00 | 250.46 | 233.41 | 240.36 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-26 11:15:00 | 285.80 | 2024-12-09 09:15:00 | 304.35 | EXIT_EMA400 | -18.55 |
| BUY | 2025-06-24 10:15:00 | 280.40 | 2025-07-23 11:15:00 | 272.70 | EXIT_EMA400 | -7.70 |
| SELL | 2025-11-18 09:15:00 | 248.78 | 2025-11-25 11:15:00 | 236.22 | TARGET | 12.56 |
| SELL | 2025-11-21 14:15:00 | 247.29 | 2025-12-01 13:15:00 | 231.49 | TARGET | 15.80 |
| SELL | 2025-11-13 11:15:00 | 246.47 | 2025-12-08 09:15:00 | 225.56 | TARGET | 20.91 |
