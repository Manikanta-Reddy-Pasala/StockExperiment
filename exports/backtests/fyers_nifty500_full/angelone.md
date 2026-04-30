# Angel One Ltd. (ANGELONE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 309.15
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / EMA400 exits:** 3 / 3
- **Total realized P&L (per unit):** 73.05
- **Avg P&L per closed trade:** 12.17

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 10:15:00 | 260.49 | 236.18 | 236.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 13:15:00 | 262.50 | 236.91 | 236.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 14:15:00 | 241.10 | 243.89 | 240.53 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-09 14:15:00 | 245.78 | 243.63 | 240.52 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 245.78 | 243.63 | 240.52 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-09-11 13:15:00 | 240.59 | 243.64 | 240.72 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 15:15:00 | 246.86 | 287.55 | 287.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 09:15:00 | 234.36 | 287.02 | 287.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 12:15:00 | 252.91 | 251.72 | 264.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-05 13:15:00 | 250.32 | 251.70 | 264.58 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 11:15:00 | 232.67 | 218.84 | 233.42 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-21 14:15:00 | 233.73 | 219.25 | 233.41 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 256.15 | 235.33 | 235.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 264.20 | 237.96 | 236.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 287.36 | 290.12 | 271.86 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 09:15:00 | 296.03 | 288.96 | 274.69 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-04 09:15:00 | 278.95 | 290.49 | 279.05 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 258.43 | 274.41 | 274.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 256.58 | 274.23 | 274.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 272.50 | 267.17 | 270.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-21 11:15:00 | 260.73 | 267.38 | 270.00 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 260.73 | 267.38 | 270.00 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-21 12:15:00 | 257.83 | 267.29 | 269.94 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 236.56 | 227.91 | 238.07 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-14 10:15:00 | 238.69 | 228.02 | 238.07 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 14:15:00 | 261.70 | 243.60 | 243.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 11:15:00 | 263.34 | 244.33 | 243.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 09:15:00 | 265.38 | 265.38 | 257.79 | EMA200 retest candle locked |

### Cycle 6 — SELL (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 10:15:00 | 234.29 | 254.90 | 254.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 09:15:00 | 233.14 | 253.72 | 254.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 249.90 | 247.20 | 250.39 | EMA200 retest candle locked |

### Cycle 7 — BUY (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 10:15:00 | 258.27 | 252.91 | 252.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 11:15:00 | 259.76 | 253.34 | 253.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 15:15:00 | 252.00 | 253.41 | 253.15 | EMA200 retest candle locked |

### Cycle 8 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 235.06 | 252.86 | 252.88 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 10:15:00 | 264.35 | 252.87 | 252.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 268.88 | 254.06 | 253.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-16 09:15:00 | 258.88 | 259.80 | 256.68 | EMA200 retest candle locked |

### Cycle 10 — SELL (started 2026-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 12:15:00 | 235.80 | 254.76 | 254.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 232.60 | 254.35 | 254.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 13:15:00 | 236.80 | 234.89 | 242.81 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-19 09:15:00 | 231.20 | 234.87 | 242.69 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-03-25 12:15:00 | 242.78 | 233.52 | 240.84 | Close above EMA400 |

### Cycle 11 — BUY (started 2026-04-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 12:15:00 | 280.89 | 244.78 | 244.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 292.59 | 246.31 | 245.53 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-09-09 14:15:00 | 245.78 | 2024-09-11 13:15:00 | 240.59 | EXIT_EMA400 | -5.19 |
| SELL | 2025-02-05 13:15:00 | 250.32 | 2025-03-03 09:15:00 | 207.55 | TARGET | 42.77 |
| BUY | 2025-06-24 09:15:00 | 296.03 | 2025-07-04 09:15:00 | 278.95 | EXIT_EMA400 | -17.08 |
| SELL | 2025-08-21 11:15:00 | 260.73 | 2025-08-28 09:15:00 | 232.93 | TARGET | 27.80 |
| SELL | 2025-08-21 12:15:00 | 257.83 | 2025-08-29 09:15:00 | 221.51 | TARGET | 36.32 |
| SELL | 2026-03-19 09:15:00 | 231.20 | 2026-03-25 12:15:00 | 242.78 | EXIT_EMA400 | -11.58 |
