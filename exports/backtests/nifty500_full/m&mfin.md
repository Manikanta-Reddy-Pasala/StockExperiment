# Mahindra & Mahindra Financial Services Ltd. (M&MFIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 310.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 4 |
| ENTRY1 | 9 |
| ENTRY2 | 0 |
| EXIT | 8 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 1
- **Winners / losers:** 1 / 7
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -37.34
- **Avg P&L per closed trade:** -4.67

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-01 11:15:00 | 287.40 | 277.71 | 277.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-02 11:15:00 | 290.05 | 278.31 | 277.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 12:15:00 | 281.95 | 282.44 | 280.34 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-13 10:15:00 | 285.25 | 282.43 | 280.39 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 11:15:00 | 283.45 | 286.36 | 283.37 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-02-28 12:15:00 | 281.30 | 286.31 | 283.36 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 09:15:00 | 259.70 | 282.04 | 282.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 13:15:00 | 259.10 | 279.77 | 280.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 11:15:00 | 277.45 | 276.77 | 279.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-27 14:15:00 | 274.60 | 276.75 | 279.03 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-03-28 10:15:00 | 279.40 | 276.76 | 279.00 | Close above EMA400 |

### Cycle 3 — BUY (started 2024-04-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 10:15:00 | 299.90 | 280.95 | 280.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 12:15:00 | 301.65 | 282.49 | 281.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 09:15:00 | 284.55 | 287.68 | 284.78 | EMA200 retest candle locked |

### Cycle 4 — SELL (started 2024-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 13:15:00 | 256.35 | 282.43 | 282.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 12:15:00 | 255.25 | 274.67 | 278.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 12:15:00 | 267.95 | 267.76 | 272.73 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-30 09:15:00 | 264.75 | 268.15 | 272.21 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-06-03 09:15:00 | 273.85 | 267.83 | 271.77 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 14:15:00 | 293.15 | 274.33 | 274.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 09:15:00 | 298.15 | 274.76 | 274.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 09:15:00 | 296.15 | 296.19 | 289.51 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-24 09:15:00 | 301.00 | 295.20 | 289.85 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 292.10 | 295.40 | 290.14 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-07-25 11:15:00 | 288.15 | 295.28 | 290.13 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 09:15:00 | 289.50 | 308.72 | 308.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 11:15:00 | 286.30 | 308.30 | 308.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 10:15:00 | 278.00 | 275.35 | 286.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-25 12:15:00 | 272.60 | 275.32 | 286.08 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-03 09:15:00 | 286.95 | 274.70 | 283.81 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 13:15:00 | 296.50 | 277.16 | 277.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 09:15:00 | 298.80 | 277.76 | 277.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-12 09:15:00 | 270.15 | 280.03 | 278.65 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-02-13 09:15:00 | 284.20 | 279.86 | 278.60 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 284.20 | 279.86 | 278.60 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-02-14 09:15:00 | 277.80 | 279.99 | 278.72 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 13:15:00 | 268.95 | 277.70 | 277.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 09:15:00 | 267.10 | 276.91 | 277.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 11:15:00 | 276.15 | 275.73 | 276.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-10 14:15:00 | 272.40 | 275.95 | 276.67 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 273.95 | 274.70 | 275.93 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-18 09:15:00 | 278.70 | 274.76 | 275.91 | Close above EMA400 |

### Cycle 9 — BUY (started 2025-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 10:15:00 | 288.30 | 276.96 | 276.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 13:15:00 | 291.45 | 277.33 | 277.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 275.80 | 280.90 | 279.15 | EMA200 retest candle locked |

### Cycle 10 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 245.50 | 277.39 | 277.53 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 13:15:00 | 272.05 | 262.82 | 262.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 276.50 | 263.97 | 263.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 11:15:00 | 272.35 | 273.04 | 268.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-30 10:15:00 | 275.80 | 273.08 | 269.00 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-09 13:15:00 | 347.90 | 368.04 | 348.54 | Close below EMA400 |

### Cycle 12 — SELL (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 12:15:00 | 317.10 | 359.54 | 359.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 314.50 | 355.53 | 357.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 314.65 | 309.59 | 324.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-28 13:15:00 | 311.90 | 310.31 | 324.50 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-02-13 10:15:00 | 285.25 | 2024-02-28 12:15:00 | 281.30 | EXIT_EMA400 | -3.95 |
| SELL | 2024-03-27 14:15:00 | 274.60 | 2024-03-28 10:15:00 | 279.40 | EXIT_EMA400 | -4.80 |
| SELL | 2024-05-30 09:15:00 | 264.75 | 2024-06-03 09:15:00 | 273.85 | EXIT_EMA400 | -9.10 |
| BUY | 2024-07-24 09:15:00 | 301.00 | 2024-07-25 11:15:00 | 288.15 | EXIT_EMA400 | -12.85 |
| SELL | 2024-11-25 12:15:00 | 272.60 | 2024-12-03 09:15:00 | 286.95 | EXIT_EMA400 | -14.35 |
| BUY | 2025-02-13 09:15:00 | 284.20 | 2025-02-14 09:15:00 | 277.80 | EXIT_EMA400 | -6.40 |
| SELL | 2025-03-10 14:15:00 | 272.40 | 2025-03-18 09:15:00 | 278.70 | EXIT_EMA400 | -6.30 |
| BUY | 2025-09-30 10:15:00 | 275.80 | 2025-10-16 09:15:00 | 296.21 | TARGET | 20.41 |
