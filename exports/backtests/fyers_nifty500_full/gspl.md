# Gujarat State Petronet Ltd. (GSPL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 285.30
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 0 / 7
- **Target hits / EMA400 exits:** 0 / 7
- **Total realized P&L (per unit):** -42.00
- **Avg P&L per closed trade:** -6.00

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 14:15:00 | 325.50 | 384.70 | 384.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 15:15:00 | 323.50 | 384.09 | 384.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 15:15:00 | 367.95 | 364.50 | 372.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-23 12:15:00 | 361.85 | 370.63 | 373.41 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-03 09:15:00 | 371.10 | 366.98 | 370.68 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 321.20 | 310.56 | 310.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-09 15:15:00 | 325.95 | 311.33 | 310.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 13:15:00 | 328.55 | 328.76 | 321.61 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-28 09:15:00 | 329.80 | 328.77 | 321.72 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 327.25 | 330.61 | 325.35 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-18 15:15:00 | 325.10 | 330.19 | 325.63 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 303.20 | 325.92 | 325.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 299.15 | 321.50 | 323.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 309.00 | 304.95 | 311.23 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 319.00 | 313.70 | 313.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 09:15:00 | 320.75 | 314.22 | 313.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 09:15:00 | 314.35 | 314.46 | 314.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-15 13:15:00 | 319.95 | 314.59 | 314.15 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 314.90 | 314.96 | 314.37 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-10-17 11:15:00 | 314.25 | 314.95 | 314.37 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 14:15:00 | 312.35 | 313.92 | 313.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 310.90 | 313.88 | 313.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 09:15:00 | 307.05 | 301.58 | 306.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-08 14:15:00 | 290.50 | 300.26 | 304.72 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-31 09:15:00 | 301.75 | 293.73 | 298.46 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 09:15:00 | 311.65 | 302.20 | 302.16 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 13:15:00 | 294.40 | 302.26 | 302.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 09:15:00 | 289.35 | 301.75 | 302.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 09:15:00 | 305.65 | 301.20 | 301.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-01 14:15:00 | 299.50 | 301.54 | 301.83 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 299.50 | 301.54 | 301.83 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-02 09:15:00 | 295.25 | 301.46 | 301.78 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 301.60 | 301.20 | 301.64 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-03 10:15:00 | 299.55 | 301.18 | 301.63 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-03 11:15:00 | 301.80 | 301.19 | 301.63 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 314.55 | 301.96 | 301.95 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 13:15:00 | 285.45 | 302.74 | 302.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 13:15:00 | 283.05 | 301.60 | 302.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 09:15:00 | 275.04 | 255.05 | 270.23 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-23 12:15:00 | 361.85 | 2025-01-03 09:15:00 | 371.10 | EXIT_EMA400 | -9.25 |
| BUY | 2025-05-28 09:15:00 | 329.80 | 2025-06-18 15:15:00 | 325.10 | EXIT_EMA400 | -4.70 |
| BUY | 2025-10-15 13:15:00 | 319.95 | 2025-10-17 11:15:00 | 314.25 | EXIT_EMA400 | -5.70 |
| SELL | 2025-12-08 14:15:00 | 290.50 | 2025-12-31 09:15:00 | 301.75 | EXIT_EMA400 | -11.25 |
| SELL | 2026-02-01 14:15:00 | 299.50 | 2026-02-03 11:15:00 | 301.80 | EXIT_EMA400 | -2.30 |
| SELL | 2026-02-02 09:15:00 | 295.25 | 2026-02-03 11:15:00 | 301.80 | EXIT_EMA400 | -6.55 |
| SELL | 2026-02-03 10:15:00 | 299.55 | 2026-02-03 11:15:00 | 301.80 | EXIT_EMA400 | -2.25 |
