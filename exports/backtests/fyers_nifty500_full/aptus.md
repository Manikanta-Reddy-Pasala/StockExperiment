# Aptus Value Housing Finance India Ltd. (APTUS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 260.45
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** 12.30
- **Avg P&L per closed trade:** 2.46

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 09:15:00 | 315.80 | 323.23 | 323.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 10:15:00 | 312.95 | 323.13 | 323.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 09:15:00 | 314.95 | 314.72 | 318.30 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-27 15:15:00 | 307.40 | 313.80 | 317.25 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 314.00 | 313.80 | 317.24 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-08-28 14:15:00 | 318.20 | 313.88 | 317.19 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 13:15:00 | 335.65 | 319.36 | 319.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 15:15:00 | 336.95 | 319.69 | 319.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 09:15:00 | 359.95 | 360.29 | 346.78 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 10:15:00 | 318.30 | 342.59 | 342.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 11:15:00 | 316.95 | 342.33 | 342.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 09:15:00 | 292.95 | 291.02 | 302.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 285.60 | 300.78 | 302.69 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 302.25 | 300.28 | 302.35 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-08 12:15:00 | 302.85 | 300.33 | 302.36 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 329.05 | 303.95 | 303.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 12:15:00 | 333.50 | 304.74 | 304.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 316.50 | 316.97 | 312.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 323.15 | 316.92 | 312.26 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-03 09:15:00 | 309.30 | 327.90 | 320.91 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 323.85 | 334.63 | 334.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 15:15:00 | 322.00 | 334.50 | 334.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 13:15:00 | 329.95 | 328.98 | 331.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-03 14:15:00 | 326.45 | 328.95 | 331.45 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 319.65 | 317.39 | 322.68 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-03 10:15:00 | 309.55 | 317.31 | 322.62 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 281.05 | 274.04 | 281.64 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-04 10:15:00 | 281.75 | 274.25 | 281.64 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-08-27 15:15:00 | 307.40 | 2024-08-28 14:15:00 | 318.20 | EXIT_EMA400 | -10.80 |
| SELL | 2025-04-07 09:15:00 | 285.60 | 2025-04-08 12:15:00 | 302.85 | EXIT_EMA400 | -17.25 |
| BUY | 2025-05-12 09:15:00 | 323.15 | 2025-06-03 09:15:00 | 309.30 | EXIT_EMA400 | -13.85 |
| SELL | 2025-10-03 14:15:00 | 326.45 | 2025-10-08 11:15:00 | 311.46 | TARGET | 14.99 |
| SELL | 2025-11-03 10:15:00 | 309.55 | 2026-01-14 14:15:00 | 270.34 | TARGET | 39.21 |
