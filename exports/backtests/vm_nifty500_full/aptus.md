# Aptus Value Housing Finance India Ltd. (APTUS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 260.14
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
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -42.55
- **Avg P&L per closed trade:** -7.09

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 14:15:00 | 311.10 | 336.18 | 336.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 15:15:00 | 310.50 | 335.92 | 336.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 13:15:00 | 328.00 | 327.37 | 331.28 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-08 10:15:00 | 325.30 | 328.92 | 331.56 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 13:15:00 | 326.35 | 328.13 | 330.94 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-04-15 09:15:00 | 318.00 | 327.90 | 330.69 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-04-23 09:15:00 | 329.50 | 324.68 | 328.48 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 09:15:00 | 346.35 | 323.11 | 323.05 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 09:15:00 | 319.25 | 326.59 | 326.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 11:15:00 | 316.70 | 326.42 | 326.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 09:15:00 | 314.95 | 314.81 | 319.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-27 15:15:00 | 307.20 | 313.86 | 318.23 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 313.95 | 313.86 | 318.21 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-08-28 14:15:00 | 318.20 | 313.93 | 318.14 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 14:15:00 | 332.75 | 320.55 | 320.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 13:15:00 | 334.15 | 321.09 | 320.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 09:15:00 | 359.90 | 360.30 | 347.04 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 10:15:00 | 318.30 | 342.61 | 342.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 11:15:00 | 316.95 | 342.35 | 342.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 09:15:00 | 292.95 | 291.00 | 302.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 285.85 | 300.74 | 302.63 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 302.25 | 300.25 | 302.30 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-08 12:15:00 | 302.85 | 300.30 | 302.30 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 328.90 | 303.92 | 303.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 10:15:00 | 329.90 | 304.17 | 303.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 316.50 | 316.96 | 312.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 323.15 | 316.91 | 312.22 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-03 09:15:00 | 309.30 | 327.87 | 320.88 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 323.85 | 334.63 | 334.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 15:15:00 | 322.45 | 334.51 | 334.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 13:15:00 | 330.00 | 328.98 | 331.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-03 14:15:00 | 326.45 | 328.95 | 331.45 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 281.05 | 274.30 | 282.05 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-04 11:15:00 | 283.30 | 274.60 | 282.05 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-04-08 10:15:00 | 325.30 | 2024-04-23 09:15:00 | 329.50 | EXIT_EMA400 | -4.20 |
| SELL | 2024-04-15 09:15:00 | 318.00 | 2024-04-23 09:15:00 | 329.50 | EXIT_EMA400 | -11.50 |
| SELL | 2024-08-27 15:15:00 | 307.20 | 2024-08-28 14:15:00 | 318.20 | EXIT_EMA400 | -11.00 |
| SELL | 2025-04-07 09:15:00 | 285.85 | 2025-04-08 12:15:00 | 302.85 | EXIT_EMA400 | -17.00 |
| BUY | 2025-05-12 09:15:00 | 323.15 | 2025-06-03 09:15:00 | 309.30 | EXIT_EMA400 | -13.85 |
| SELL | 2025-10-03 14:15:00 | 326.45 | 2025-10-08 11:15:00 | 311.45 | TARGET | 15.00 |
