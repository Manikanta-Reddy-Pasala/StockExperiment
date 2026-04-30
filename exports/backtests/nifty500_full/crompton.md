# Crompton Greaves Consumer Electricals Ltd. (CROMPTON.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 272.36
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** 20.72
- **Avg P&L per closed trade:** 2.96

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 15:15:00 | 284.80 | 297.74 | 297.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-26 09:15:00 | 282.25 | 297.59 | 297.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-17 10:15:00 | 288.10 | 287.99 | 291.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-17 11:15:00 | 286.80 | 287.98 | 291.61 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 289.25 | 287.89 | 291.35 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-11-22 09:15:00 | 292.55 | 288.10 | 291.34 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 12:15:00 | 305.45 | 292.27 | 292.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-29 11:15:00 | 310.90 | 294.91 | 293.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 308.75 | 309.61 | 303.08 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-13 09:15:00 | 284.25 | 300.63 | 300.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-15 11:15:00 | 281.30 | 298.23 | 299.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-23 13:15:00 | 295.50 | 295.07 | 297.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-26 09:15:00 | 290.70 | 294.99 | 297.34 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-03-01 13:15:00 | 296.55 | 293.99 | 296.44 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 14:15:00 | 311.90 | 289.88 | 289.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 10:15:00 | 315.85 | 291.92 | 290.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 424.20 | 428.58 | 406.75 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-07 12:15:00 | 433.40 | 428.25 | 408.25 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-09-24 10:15:00 | 439.00 | 452.94 | 440.79 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 12:15:00 | 409.85 | 434.72 | 434.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 403.65 | 433.16 | 433.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 11:15:00 | 401.35 | 399.16 | 410.55 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-17 15:15:00 | 396.55 | 405.57 | 409.71 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-12 10:15:00 | 355.50 | 342.55 | 354.87 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 347.85 | 345.31 | 345.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 11:15:00 | 350.00 | 345.39 | 345.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 10:15:00 | 347.45 | 347.48 | 346.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 13:15:00 | 349.00 | 347.04 | 346.33 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 349.00 | 347.04 | 346.33 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-16 15:15:00 | 349.55 | 347.09 | 346.36 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 347.25 | 347.29 | 346.50 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-18 13:15:00 | 346.05 | 347.27 | 346.50 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 334.70 | 346.86 | 346.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 11:15:00 | 333.10 | 346.60 | 346.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 14:15:00 | 329.85 | 329.41 | 335.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-20 09:15:00 | 327.30 | 329.39 | 335.51 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-29 10:15:00 | 333.60 | 327.39 | 333.21 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 12:15:00 | 278.65 | 250.67 | 250.56 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-17 11:15:00 | 286.80 | 2023-11-22 09:15:00 | 292.55 | EXIT_EMA400 | -5.75 |
| SELL | 2024-02-26 09:15:00 | 290.70 | 2024-03-01 13:15:00 | 296.55 | EXIT_EMA400 | -5.85 |
| BUY | 2024-08-07 12:15:00 | 433.40 | 2024-09-24 10:15:00 | 439.00 | EXIT_EMA400 | 5.60 |
| SELL | 2024-12-17 15:15:00 | 396.55 | 2025-01-13 09:15:00 | 357.08 | TARGET | 39.47 |
| BUY | 2025-06-16 13:15:00 | 349.00 | 2025-06-18 13:15:00 | 346.05 | EXIT_EMA400 | -2.95 |
| BUY | 2025-06-16 15:15:00 | 349.55 | 2025-06-18 13:15:00 | 346.05 | EXIT_EMA400 | -3.50 |
| SELL | 2025-08-20 09:15:00 | 327.30 | 2025-08-29 10:15:00 | 333.60 | EXIT_EMA400 | -6.30 |
