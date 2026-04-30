# Angel One Ltd. (ANGELONE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 308.71
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT3 | 7 |
| ENTRY1 | 6 |
| ENTRY2 | 4 |
| EXIT | 6 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 7 / 3
- **Target hits / EMA400 exits:** 6 / 4
- **Total realized P&L (per unit):** 165.00
- **Avg P&L per closed trade:** 16.50

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 10:15:00 | 287.11 | 310.98 | 311.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 280.91 | 309.58 | 310.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 09:15:00 | 287.71 | 281.01 | 292.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-06 09:15:00 | 264.49 | 283.95 | 289.14 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 268.47 | 259.48 | 269.06 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-06-12 11:15:00 | 262.68 | 259.71 | 268.98 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 265.90 | 259.91 | 268.85 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-06-13 10:15:00 | 262.95 | 259.94 | 268.82 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 11:15:00 | 233.06 | 221.25 | 233.46 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-08-19 12:15:00 | 236.56 | 221.40 | 233.48 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 11:15:00 | 257.25 | 241.97 | 241.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 13:15:00 | 259.39 | 243.84 | 243.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 09:15:00 | 247.96 | 249.57 | 246.49 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-27 13:15:00 | 254.60 | 249.44 | 246.59 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 274.05 | 251.28 | 247.77 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-03 10:15:00 | 277.37 | 251.54 | 247.92 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-11-12 15:15:00 | 272.80 | 284.02 | 273.35 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 15:15:00 | 246.86 | 287.54 | 287.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 09:15:00 | 234.39 | 287.01 | 287.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 12:15:00 | 252.90 | 252.63 | 265.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-05 13:15:00 | 250.31 | 252.60 | 265.51 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 233.73 | 219.36 | 233.73 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-21 15:15:00 | 234.00 | 219.50 | 233.73 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 11:15:00 | 256.48 | 235.55 | 235.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 15:15:00 | 257.09 | 237.71 | 236.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 287.36 | 290.13 | 271.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 09:15:00 | 296.03 | 288.98 | 274.73 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-04 09:15:00 | 278.95 | 290.50 | 279.09 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 258.43 | 274.41 | 274.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 256.58 | 274.23 | 274.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 272.50 | 267.18 | 270.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-21 11:15:00 | 260.73 | 267.38 | 270.01 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 260.73 | 267.38 | 270.01 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-21 12:15:00 | 257.83 | 267.28 | 269.95 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 236.56 | 227.90 | 238.06 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-14 10:15:00 | 238.80 | 228.01 | 238.07 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 14:15:00 | 261.57 | 243.59 | 243.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 11:15:00 | 263.34 | 244.33 | 243.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 10:15:00 | 264.82 | 265.37 | 257.81 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 10:15:00 | 234.29 | 254.90 | 254.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 09:15:00 | 233.14 | 253.73 | 254.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 250.05 | 247.22 | 250.39 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 10:15:00 | 258.27 | 252.91 | 252.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 11:15:00 | 259.50 | 253.33 | 253.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 15:15:00 | 252.18 | 253.41 | 253.16 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 230.60 | 252.78 | 252.84 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 11:15:00 | 267.35 | 253.01 | 252.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 14:15:00 | 268.89 | 253.45 | 253.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-16 09:15:00 | 258.88 | 260.38 | 257.09 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2026-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 12:15:00 | 235.50 | 255.07 | 255.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 232.80 | 254.66 | 254.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 13:15:00 | 236.80 | 235.04 | 243.02 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-19 09:15:00 | 231.20 | 235.02 | 242.90 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-03-25 12:15:00 | 242.80 | 233.63 | 241.02 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 13:15:00 | 280.90 | 245.18 | 245.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 292.60 | 246.35 | 245.65 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-06-12 11:15:00 | 262.68 | 2024-07-02 09:15:00 | 243.78 | TARGET | 18.90 |
| SELL | 2024-06-13 10:15:00 | 262.95 | 2024-07-02 09:15:00 | 245.31 | TARGET | 17.64 |
| SELL | 2024-05-06 09:15:00 | 264.49 | 2024-08-19 12:15:00 | 236.56 | EXIT_EMA400 | 27.93 |
| BUY | 2024-09-27 13:15:00 | 254.60 | 2024-10-03 10:15:00 | 278.63 | TARGET | 24.03 |
| BUY | 2024-10-03 10:15:00 | 277.37 | 2024-11-12 15:15:00 | 272.80 | EXIT_EMA400 | -4.57 |
| SELL | 2025-02-05 13:15:00 | 250.31 | 2025-03-03 09:15:00 | 204.73 | TARGET | 45.58 |
| BUY | 2025-06-24 09:15:00 | 296.03 | 2025-07-04 09:15:00 | 278.95 | EXIT_EMA400 | -17.08 |
| SELL | 2025-08-21 11:15:00 | 260.73 | 2025-08-28 09:15:00 | 232.90 | TARGET | 27.83 |
| SELL | 2025-08-21 12:15:00 | 257.83 | 2025-08-29 09:15:00 | 221.48 | TARGET | 36.35 |
| SELL | 2026-03-19 09:15:00 | 231.20 | 2026-03-25 12:15:00 | 242.80 | EXIT_EMA400 | -11.60 |
