# Hindustan Copper Ltd. (HINDCOPPER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 538.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 2 / 2
- **Total realized P&L (per unit):** 26.43
- **Avg P&L per closed trade:** 6.61

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 09:15:00 | 349.65 | 324.30 | 324.22 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 09:15:00 | 319.00 | 324.70 | 324.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 10:15:00 | 310.60 | 321.92 | 323.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 13:15:00 | 285.90 | 283.44 | 295.38 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-17 15:15:00 | 281.75 | 285.67 | 293.07 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 11:15:00 | 232.65 | 222.12 | 232.98 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-20 12:15:00 | 234.76 | 222.25 | 232.99 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 14:15:00 | 245.65 | 224.23 | 224.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 253.69 | 224.72 | 224.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 243.69 | 245.41 | 237.67 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-19 13:15:00 | 245.95 | 245.42 | 237.71 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-25 15:15:00 | 256.20 | 265.47 | 256.51 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 239.80 | 251.54 | 251.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 15:15:00 | 239.11 | 248.79 | 250.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 243.80 | 242.87 | 246.47 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 283.10 | 248.59 | 248.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 11:15:00 | 289.76 | 249.37 | 248.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 325.45 | 329.01 | 307.54 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-10 09:15:00 | 337.70 | 328.87 | 308.92 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-21 14:15:00 | 314.25 | 331.14 | 316.22 | Close below EMA400 |

### Cycle 6 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 458.75 | 516.63 | 516.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 455.00 | 516.01 | 516.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 526.75 | 511.34 | 513.83 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 09:15:00 | 519.95 | 514.37 | 515.17 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 519.95 | 514.37 | 515.17 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-13 10:15:00 | 528.75 | 514.51 | 515.24 | Close above EMA400 |

### Cycle 7 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 554.40 | 516.04 | 515.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 10:15:00 | 561.40 | 518.62 | 517.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 14:15:00 | 534.40 | 537.37 | 528.93 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-17 15:15:00 | 281.75 | 2024-12-30 14:15:00 | 247.78 | TARGET | 33.97 |
| BUY | 2025-06-19 13:15:00 | 245.95 | 2025-06-26 14:15:00 | 270.66 | TARGET | 24.71 |
| BUY | 2025-11-10 09:15:00 | 337.70 | 2025-11-21 14:15:00 | 314.25 | EXIT_EMA400 | -23.45 |
| SELL | 2026-04-13 09:15:00 | 519.95 | 2026-04-13 10:15:00 | 528.75 | EXIT_EMA400 | -8.80 |
