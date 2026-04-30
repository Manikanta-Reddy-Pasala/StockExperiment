# RITES Ltd. (RITES.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 218.91
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 4 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / EMA400 exits:** 4 / 2
- **Total realized P&L (per unit):** 73.94
- **Avg P&L per closed trade:** 12.32

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-06 13:15:00 | 224.50 | 236.22 | 236.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 09:15:00 | 223.62 | 234.48 | 235.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-17 12:15:00 | 239.77 | 231.36 | 233.45 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 12:15:00 | 242.35 | 234.70 | 234.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 13:15:00 | 245.25 | 235.54 | 235.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 14:15:00 | 242.38 | 243.30 | 239.56 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-26 09:15:00 | 256.30 | 243.35 | 239.87 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-13 09:15:00 | 332.85 | 362.65 | 334.80 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 12:15:00 | 330.48 | 350.84 | 350.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-16 13:15:00 | 328.48 | 350.62 | 350.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 339.15 | 336.59 | 342.15 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 12:15:00 | 359.95 | 344.54 | 344.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 13:15:00 | 361.15 | 344.71 | 344.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 09:15:00 | 345.85 | 347.62 | 346.12 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 10:15:00 | 321.30 | 344.81 | 344.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 12:15:00 | 317.55 | 344.30 | 344.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 301.40 | 292.05 | 307.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-26 10:15:00 | 287.25 | 292.08 | 306.61 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-09 09:15:00 | 304.35 | 290.73 | 301.93 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 11:15:00 | 296.33 | 235.00 | 234.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 305.30 | 262.49 | 251.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 276.45 | 278.14 | 264.25 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 10:15:00 | 280.25 | 277.29 | 265.18 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 273.15 | 278.93 | 272.81 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-23 11:15:00 | 272.70 | 278.81 | 272.81 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 252.25 | 269.10 | 269.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 248.70 | 268.53 | 268.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 14:15:00 | 258.40 | 257.40 | 261.70 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 15:15:00 | 270.56 | 263.93 | 263.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 11:15:00 | 271.50 | 264.12 | 264.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 263.28 | 264.57 | 264.25 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 10:15:00 | 255.91 | 263.89 | 263.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 11:15:00 | 254.79 | 263.80 | 263.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 11:15:00 | 249.71 | 249.39 | 253.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-13 11:15:00 | 246.47 | 249.33 | 253.44 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 252.78 | 248.99 | 253.01 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-18 09:15:00 | 248.78 | 249.15 | 252.97 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 252.35 | 249.24 | 252.64 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-21 14:15:00 | 247.29 | 249.25 | 252.56 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 238.31 | 233.19 | 240.33 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-23 11:15:00 | 250.56 | 233.42 | 240.37 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-12-26 09:15:00 | 256.30 | 2024-01-23 09:15:00 | 305.59 | TARGET | 49.29 |
| SELL | 2024-11-26 10:15:00 | 287.25 | 2024-12-09 09:15:00 | 304.35 | EXIT_EMA400 | -17.10 |
| BUY | 2025-06-24 10:15:00 | 280.25 | 2025-07-23 11:15:00 | 272.70 | EXIT_EMA400 | -7.55 |
| SELL | 2025-11-18 09:15:00 | 248.78 | 2025-11-25 11:15:00 | 236.21 | TARGET | 12.57 |
| SELL | 2025-11-21 14:15:00 | 247.29 | 2025-12-01 13:15:00 | 231.48 | TARGET | 15.81 |
| SELL | 2025-11-13 11:15:00 | 246.47 | 2025-12-08 09:15:00 | 225.55 | TARGET | 20.92 |
