# AWL Agri Business Ltd. (AWL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 196.37
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| EXIT | 7 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** -54.34
- **Avg P&L per closed trade:** -7.76

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 09:15:00 | 372.85 | 343.27 | 343.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-03 09:15:00 | 394.30 | 356.01 | 351.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-15 15:15:00 | 363.30 | 363.47 | 357.03 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2024-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 15:15:00 | 338.25 | 354.41 | 354.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-13 09:15:00 | 334.60 | 354.22 | 354.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-16 12:15:00 | 350.80 | 350.71 | 352.50 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2024-02-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 10:15:00 | 360.95 | 354.06 | 354.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 09:15:00 | 384.05 | 354.68 | 354.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 14:15:00 | 365.40 | 366.89 | 361.50 | EMA200 retest candle locked |

### Cycle 4 — SELL (started 2024-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 12:15:00 | 337.10 | 357.94 | 357.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 14:15:00 | 335.25 | 357.49 | 357.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-02 11:15:00 | 352.90 | 347.45 | 351.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-15 10:15:00 | 338.50 | 349.34 | 352.02 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-30 09:15:00 | 355.35 | 345.36 | 348.83 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-05 12:15:00 | 377.80 | 340.48 | 340.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-05 13:15:00 | 387.45 | 340.95 | 340.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 10:15:00 | 354.35 | 354.40 | 348.25 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-16 12:15:00 | 362.15 | 354.47 | 348.55 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 358.60 | 365.28 | 358.02 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-09-09 12:15:00 | 357.50 | 365.09 | 358.03 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 13:15:00 | 339.55 | 355.29 | 355.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 13:15:00 | 338.35 | 354.28 | 354.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 14:15:00 | 339.50 | 338.47 | 345.13 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-21 09:15:00 | 294.90 | 334.52 | 339.92 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 320.05 | 311.75 | 320.71 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-26 14:15:00 | 321.00 | 311.84 | 320.71 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 14:15:00 | 283.15 | 270.05 | 270.02 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 10:15:00 | 263.45 | 270.11 | 270.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 15:15:00 | 261.45 | 269.75 | 269.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-14 09:15:00 | 267.40 | 267.32 | 268.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-20 13:15:00 | 264.55 | 267.75 | 268.64 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-27 11:15:00 | 272.90 | 265.64 | 267.36 | Close above EMA400 |

### Cycle 9 — BUY (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 14:15:00 | 278.10 | 265.49 | 265.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 15:15:00 | 280.00 | 265.64 | 265.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 12:15:00 | 267.50 | 268.35 | 267.02 | EMA200 retest candle locked |

### Cycle 10 — SELL (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 12:15:00 | 253.45 | 265.98 | 266.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 13:15:00 | 252.95 | 265.85 | 265.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 15:15:00 | 260.20 | 259.73 | 262.34 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-22 09:15:00 | 258.75 | 259.87 | 262.23 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 260.10 | 257.39 | 260.33 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-03 11:15:00 | 263.85 | 257.46 | 260.34 | Close above EMA400 |

### Cycle 11 — BUY (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 09:15:00 | 263.05 | 261.08 | 261.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 11:15:00 | 265.10 | 261.14 | 261.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 14:15:00 | 261.75 | 263.21 | 262.27 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-27 14:15:00 | 265.85 | 262.80 | 262.17 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 269.50 | 269.79 | 266.94 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-25 09:15:00 | 264.10 | 270.15 | 267.32 | Close below EMA400 |

### Cycle 12 — SELL (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 10:15:00 | 246.40 | 265.17 | 265.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 11:15:00 | 245.35 | 261.10 | 263.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-11 11:15:00 | 196.40 | 196.30 | 210.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-12 09:15:00 | 176.50 | 195.55 | 210.01 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-22 09:15:00 | 196.35 | 184.19 | 193.40 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-04-15 10:15:00 | 338.50 | 2024-04-30 09:15:00 | 355.35 | EXIT_EMA400 | -16.85 |
| BUY | 2024-08-16 12:15:00 | 362.15 | 2024-09-09 12:15:00 | 357.50 | EXIT_EMA400 | -4.65 |
| SELL | 2024-11-21 09:15:00 | 294.90 | 2024-12-26 14:15:00 | 321.00 | EXIT_EMA400 | -26.10 |
| SELL | 2025-05-20 13:15:00 | 264.55 | 2025-05-27 11:15:00 | 272.90 | EXIT_EMA400 | -8.35 |
| SELL | 2025-08-22 09:15:00 | 258.75 | 2025-08-28 09:15:00 | 248.32 | TARGET | 10.43 |
| BUY | 2025-10-27 14:15:00 | 265.85 | 2025-11-04 14:15:00 | 276.89 | TARGET | 11.04 |
| SELL | 2026-03-12 09:15:00 | 176.50 | 2026-04-22 09:15:00 | 196.35 | EXIT_EMA400 | -19.85 |
