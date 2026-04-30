# Bharat Petroleum Corporation Ltd. (BPCL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 300.45
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 5 |
| ENTRY1 | 6 |
| ENTRY2 | 4 |
| EXIT | 5 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 1
- **Winners / losers:** 5 / 4
- **Target hits / EMA400 exits:** 5 / 4
- **Total realized P&L (per unit):** 125.58
- **Avg P&L per closed trade:** 13.95

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-09 11:15:00 | 191.95 | 177.59 | 177.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-13 10:15:00 | 194.27 | 179.40 | 178.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 14:15:00 | 297.10 | 300.72 | 277.79 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-03-14 14:15:00 | 304.30 | 300.78 | 278.61 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-19 14:15:00 | 278.60 | 298.30 | 279.52 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 11:15:00 | 310.85 | 336.28 | 336.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 13:15:00 | 308.55 | 335.75 | 336.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 12:15:00 | 303.85 | 303.70 | 313.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-13 10:15:00 | 298.10 | 303.64 | 312.32 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 267.35 | 257.75 | 267.50 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-12 11:15:00 | 264.27 | 257.90 | 267.48 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 265.12 | 259.42 | 266.84 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-20 10:15:00 | 270.96 | 259.53 | 266.86 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 14:15:00 | 286.05 | 271.60 | 271.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 11:15:00 | 289.50 | 272.14 | 271.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 14:15:00 | 309.90 | 310.83 | 300.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-04 11:15:00 | 311.75 | 310.80 | 300.99 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 309.45 | 314.25 | 304.91 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-13 11:15:00 | 313.15 | 314.19 | 304.97 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-31 09:15:00 | 326.90 | 336.27 | 327.11 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 310.55 | 322.58 | 322.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 308.45 | 322.33 | 322.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 322.90 | 318.94 | 320.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-12 15:15:00 | 318.00 | 319.11 | 320.46 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 318.80 | 319.11 | 320.45 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-15 10:15:00 | 317.85 | 319.09 | 320.43 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-16 09:15:00 | 320.70 | 319.03 | 320.36 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 14:15:00 | 330.65 | 321.42 | 321.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 09:15:00 | 335.20 | 322.85 | 322.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 13:15:00 | 331.55 | 332.17 | 327.90 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-15 10:15:00 | 338.95 | 332.25 | 328.02 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 334.00 | 333.32 | 329.31 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-27 09:15:00 | 339.85 | 333.28 | 329.43 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-12-11 09:15:00 | 350.70 | 358.14 | 351.27 | Close below EMA400 |

### Cycle 6 — SELL (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 09:15:00 | 325.15 | 364.42 | 364.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 10:15:00 | 323.90 | 364.02 | 364.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 14:15:00 | 310.30 | 307.33 | 325.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-24 09:15:00 | 301.90 | 309.08 | 323.56 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-03-14 14:15:00 | 304.30 | 2024-03-19 14:15:00 | 278.60 | EXIT_EMA400 | -25.70 |
| SELL | 2024-12-13 10:15:00 | 298.10 | 2025-01-28 10:15:00 | 255.45 | TARGET | 42.65 |
| SELL | 2025-03-12 11:15:00 | 264.27 | 2025-03-20 10:15:00 | 270.96 | EXIT_EMA400 | -6.69 |
| BUY | 2025-06-13 11:15:00 | 313.15 | 2025-07-04 11:15:00 | 337.68 | TARGET | 24.53 |
| BUY | 2025-06-04 11:15:00 | 311.75 | 2025-07-04 14:15:00 | 344.03 | TARGET | 32.28 |
| SELL | 2025-09-12 15:15:00 | 318.00 | 2025-09-16 09:15:00 | 320.70 | EXIT_EMA400 | -2.70 |
| SELL | 2025-09-15 10:15:00 | 317.85 | 2025-09-16 09:15:00 | 320.70 | EXIT_EMA400 | -2.85 |
| BUY | 2025-10-15 10:15:00 | 338.95 | 2025-11-04 09:15:00 | 371.74 | TARGET | 32.79 |
| BUY | 2025-10-27 09:15:00 | 339.85 | 2025-11-04 09:15:00 | 371.12 | TARGET | 31.27 |
