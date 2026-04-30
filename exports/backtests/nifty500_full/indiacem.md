# India Cements Ltd. (INDIACEM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 395.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 1 / 7
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -52.00
- **Avg P&L per closed trade:** -6.50

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 10:15:00 | 216.95 | 230.36 | 230.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 11:15:00 | 215.25 | 230.21 | 230.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-16 12:15:00 | 219.35 | 217.63 | 222.08 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 10:15:00 | 259.90 | 224.49 | 224.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 13:15:00 | 263.05 | 230.67 | 227.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 15:15:00 | 247.00 | 247.29 | 238.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-21 09:15:00 | 249.40 | 247.31 | 238.25 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-01-18 09:15:00 | 246.85 | 255.93 | 248.39 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-02-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 15:15:00 | 235.45 | 245.80 | 245.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 13:15:00 | 234.40 | 244.53 | 245.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 12:15:00 | 219.45 | 219.34 | 228.49 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-06 12:15:00 | 217.65 | 223.81 | 226.65 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 12:15:00 | 217.75 | 213.50 | 218.40 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-06-04 09:15:00 | 205.85 | 213.49 | 218.30 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 11:15:00 | 213.75 | 210.45 | 216.13 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-06-07 12:15:00 | 212.80 | 210.47 | 216.12 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-06-10 09:15:00 | 222.10 | 210.70 | 216.12 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 10:15:00 | 263.78 | 219.40 | 219.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 11:15:00 | 264.77 | 219.86 | 219.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 14:15:00 | 362.05 | 362.18 | 342.02 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-23 12:15:00 | 367.25 | 362.36 | 342.61 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-25 10:15:00 | 352.10 | 361.57 | 353.10 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 09:15:00 | 332.65 | 354.83 | 354.86 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 376.45 | 354.53 | 354.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 09:15:00 | 378.00 | 355.86 | 355.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 368.25 | 370.41 | 364.72 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 12:15:00 | 297.70 | 359.72 | 359.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 276.10 | 357.01 | 358.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 13:15:00 | 286.50 | 280.99 | 304.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-11 14:15:00 | 278.85 | 284.03 | 302.42 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 12:15:00 | 287.95 | 278.41 | 288.60 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-16 13:15:00 | 283.20 | 278.46 | 288.57 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-04-17 09:15:00 | 294.55 | 278.71 | 288.54 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 15:15:00 | 309.40 | 293.18 | 293.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 314.75 | 294.32 | 293.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 15:15:00 | 312.75 | 315.31 | 306.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-02 09:15:00 | 325.00 | 315.41 | 306.94 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 319.80 | 327.12 | 317.62 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-20 09:15:00 | 316.30 | 326.85 | 317.62 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 13:15:00 | 388.80 | 432.19 | 432.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 12:15:00 | 386.40 | 427.06 | 429.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 376.75 | 375.99 | 394.70 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-12-21 09:15:00 | 249.40 | 2024-01-18 09:15:00 | 246.85 | EXIT_EMA400 | -2.55 |
| SELL | 2024-05-06 12:15:00 | 217.65 | 2024-06-04 11:15:00 | 190.65 | TARGET | 27.00 |
| SELL | 2024-06-04 09:15:00 | 205.85 | 2024-06-10 09:15:00 | 222.10 | EXIT_EMA400 | -16.25 |
| SELL | 2024-06-07 12:15:00 | 212.80 | 2024-06-10 09:15:00 | 222.10 | EXIT_EMA400 | -9.30 |
| BUY | 2024-09-23 12:15:00 | 367.25 | 2024-10-25 10:15:00 | 352.10 | EXIT_EMA400 | -15.15 |
| SELL | 2025-03-11 14:15:00 | 278.85 | 2025-04-17 09:15:00 | 294.55 | EXIT_EMA400 | -15.70 |
| SELL | 2025-04-16 13:15:00 | 283.20 | 2025-04-17 09:15:00 | 294.55 | EXIT_EMA400 | -11.35 |
| BUY | 2025-06-02 09:15:00 | 325.00 | 2025-06-20 09:15:00 | 316.30 | EXIT_EMA400 | -8.70 |
