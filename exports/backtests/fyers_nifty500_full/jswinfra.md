# JSW Infrastructure Ltd. (JSWINFRA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 272.80
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
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 1
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -25.57
- **Avg P&L per closed trade:** -4.26

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 09:15:00 | 292.75 | 321.58 | 321.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 14:15:00 | 289.25 | 320.09 | 320.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 14:15:00 | 316.00 | 314.93 | 318.13 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-31 13:15:00 | 312.40 | 315.11 | 318.02 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-01 17:15:00 | 323.15 | 315.18 | 318.02 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 14:15:00 | 328.60 | 315.27 | 315.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 15:15:00 | 329.35 | 315.41 | 315.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 15:15:00 | 317.20 | 317.29 | 316.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-18 09:15:00 | 319.45 | 317.31 | 316.38 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 319.45 | 317.31 | 316.38 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-19 09:15:00 | 314.60 | 317.35 | 316.43 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 09:15:00 | 293.80 | 316.23 | 316.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 15:15:00 | 288.45 | 308.95 | 312.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 09:15:00 | 259.45 | 259.23 | 277.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-03 09:15:00 | 247.60 | 258.27 | 274.65 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 270.60 | 256.99 | 271.79 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-07 10:15:00 | 271.95 | 257.14 | 271.79 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 13:15:00 | 323.55 | 278.27 | 278.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 09:15:00 | 326.75 | 309.63 | 304.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 10:15:00 | 308.80 | 311.32 | 305.90 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 14:15:00 | 300.00 | 304.29 | 304.29 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 11:15:00 | 310.30 | 304.31 | 304.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 12:15:00 | 312.85 | 304.40 | 304.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 11:15:00 | 318.30 | 319.78 | 313.51 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-29 14:15:00 | 324.25 | 319.81 | 313.62 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-30 11:15:00 | 313.60 | 319.71 | 313.69 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 11:15:00 | 297.65 | 310.74 | 310.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 295.00 | 310.02 | 310.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 11:15:00 | 275.25 | 274.89 | 283.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 14:15:00 | 272.40 | 278.92 | 282.92 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 275.95 | 274.38 | 279.68 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-19 14:15:00 | 272.50 | 274.36 | 279.64 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-03-06 09:15:00 | 268.80 | 259.41 | 265.47 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 14:15:00 | 274.71 | 261.75 | 261.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 09:15:00 | 279.20 | 262.05 | 261.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 266.01 | 266.66 | 264.39 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-30 13:15:00 | 273.98 | 266.77 | 264.49 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-31 13:15:00 | 312.40 | 2024-11-01 17:15:00 | 323.15 | EXIT_EMA400 | -10.75 |
| BUY | 2024-12-18 09:15:00 | 319.45 | 2024-12-19 09:15:00 | 314.60 | EXIT_EMA400 | -4.85 |
| SELL | 2025-03-03 09:15:00 | 247.60 | 2025-03-07 10:15:00 | 271.95 | EXIT_EMA400 | -24.35 |
| BUY | 2025-09-29 14:15:00 | 324.25 | 2025-09-30 11:15:00 | 313.60 | EXIT_EMA400 | -10.65 |
| SELL | 2026-01-19 14:15:00 | 272.50 | 2026-03-02 09:15:00 | 251.07 | TARGET | 21.43 |
| SELL | 2026-01-08 14:15:00 | 272.40 | 2026-03-06 09:15:00 | 268.80 | EXIT_EMA400 | 3.60 |
