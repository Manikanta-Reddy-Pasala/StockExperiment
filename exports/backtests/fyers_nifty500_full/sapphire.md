# Sapphire Foods India Ltd. (SAPPHIRE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 204.89
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** -6.64
- **Avg P&L per closed trade:** -0.95

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 318.90 | 333.53 | 333.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 312.80 | 332.75 | 333.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 09:15:00 | 321.65 | 320.39 | 325.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-13 09:15:00 | 312.50 | 327.10 | 328.01 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 324.40 | 321.77 | 324.86 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-23 13:15:00 | 319.00 | 321.74 | 324.83 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-12-24 09:15:00 | 326.15 | 321.74 | 324.79 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 12:15:00 | 355.00 | 326.82 | 326.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-06 09:15:00 | 365.50 | 327.81 | 327.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 13:15:00 | 329.15 | 332.56 | 330.02 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 12:15:00 | 314.40 | 328.04 | 328.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 312.45 | 327.75 | 327.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 13:15:00 | 316.20 | 313.61 | 319.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-07 09:15:00 | 300.00 | 315.13 | 319.82 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 300.00 | 315.13 | 319.82 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-02-07 15:15:00 | 323.80 | 315.19 | 319.71 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 12:15:00 | 321.10 | 311.20 | 311.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 323.80 | 311.57 | 311.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 309.10 | 313.04 | 312.16 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-12 10:15:00 | 308.00 | 311.38 | 311.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 11:15:00 | 303.50 | 311.03 | 311.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 09:15:00 | 310.55 | 310.21 | 310.76 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 09:15:00 | 335.65 | 311.50 | 311.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 09:15:00 | 349.40 | 323.00 | 319.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 11:15:00 | 326.60 | 327.19 | 322.67 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-11 12:15:00 | 329.20 | 327.21 | 322.71 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 330.90 | 331.17 | 326.05 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-23 14:15:00 | 342.95 | 331.28 | 326.13 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-28 09:15:00 | 326.60 | 331.88 | 326.84 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 09:15:00 | 318.05 | 323.86 | 323.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 310.00 | 323.25 | 323.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 13:15:00 | 322.65 | 321.42 | 322.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-28 11:15:00 | 320.30 | 322.96 | 323.21 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 319.80 | 322.83 | 323.14 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-08-29 10:15:00 | 325.15 | 322.84 | 323.14 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 15:15:00 | 325.75 | 323.41 | 323.40 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 321.65 | 323.39 | 323.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 321.10 | 323.37 | 323.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 333.15 | 323.00 | 323.18 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 13:15:00 | 333.10 | 323.39 | 323.38 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 311.90 | 323.52 | 323.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 13:15:00 | 309.05 | 323.00 | 323.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 09:15:00 | 297.65 | 294.71 | 303.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-31 10:15:00 | 290.60 | 295.09 | 303.83 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 254.40 | 247.59 | 260.14 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-01 14:15:00 | 261.95 | 248.05 | 260.06 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-13 09:15:00 | 312.50 | 2024-12-24 09:15:00 | 326.15 | EXIT_EMA400 | -13.65 |
| SELL | 2024-12-23 13:15:00 | 319.00 | 2024-12-24 09:15:00 | 326.15 | EXIT_EMA400 | -7.15 |
| SELL | 2025-02-07 09:15:00 | 300.00 | 2025-02-07 15:15:00 | 323.80 | EXIT_EMA400 | -23.80 |
| BUY | 2025-07-11 12:15:00 | 329.20 | 2025-07-17 12:15:00 | 348.68 | TARGET | 19.48 |
| BUY | 2025-07-23 14:15:00 | 342.95 | 2025-07-28 09:15:00 | 326.60 | EXIT_EMA400 | -16.35 |
| SELL | 2025-08-28 11:15:00 | 320.30 | 2025-08-29 10:15:00 | 325.15 | EXIT_EMA400 | -4.85 |
| SELL | 2025-10-31 10:15:00 | 290.60 | 2025-11-13 09:15:00 | 250.92 | TARGET | 39.68 |
