# Bharat Petroleum Corporation Ltd. (BPCL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 301.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 1
- **Winners / losers:** 5 / 1
- **Target hits / EMA400 exits:** 5 / 1
- **Total realized P&L (per unit):** 155.95
- **Avg P&L per closed trade:** 25.99

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 10:15:00 | 310.85 | 336.54 | 336.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 10:15:00 | 304.65 | 334.68 | 335.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 12:15:00 | 303.75 | 303.64 | 313.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-13 10:15:00 | 298.10 | 303.58 | 312.26 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-12 09:15:00 | 267.35 | 257.55 | 267.08 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 11:15:00 | 281.40 | 271.15 | 271.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 12:15:00 | 283.00 | 271.26 | 271.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 14:15:00 | 309.90 | 310.81 | 300.73 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-04 11:15:00 | 311.75 | 310.79 | 300.92 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 309.45 | 314.25 | 304.86 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-13 11:15:00 | 313.30 | 314.19 | 304.92 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-31 09:15:00 | 326.90 | 336.28 | 327.10 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 310.55 | 322.59 | 322.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 308.45 | 322.34 | 322.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 322.85 | 318.95 | 320.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-15 11:15:00 | 317.45 | 319.09 | 320.42 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-16 09:15:00 | 320.70 | 319.05 | 320.37 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 14:15:00 | 330.65 | 321.44 | 321.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 09:15:00 | 335.30 | 322.86 | 322.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 12:15:00 | 332.15 | 332.18 | 327.88 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-15 10:15:00 | 338.95 | 332.25 | 328.02 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 334.00 | 333.32 | 329.31 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-27 09:15:00 | 338.20 | 333.29 | 329.43 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-12-11 09:15:00 | 350.90 | 358.14 | 351.27 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 09:15:00 | 325.15 | 364.39 | 364.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 10:15:00 | 323.85 | 363.99 | 364.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 14:15:00 | 310.40 | 307.34 | 325.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-24 09:15:00 | 301.90 | 309.28 | 323.38 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-13 10:15:00 | 298.10 | 2025-01-28 09:15:00 | 255.61 | TARGET | 42.49 |
| BUY | 2025-06-13 11:15:00 | 313.30 | 2025-07-04 11:15:00 | 338.43 | TARGET | 25.13 |
| BUY | 2025-06-04 11:15:00 | 311.75 | 2025-07-04 14:15:00 | 344.24 | TARGET | 32.49 |
| SELL | 2025-09-15 11:15:00 | 317.45 | 2025-09-16 09:15:00 | 320.70 | EXIT_EMA400 | -3.25 |
| BUY | 2025-10-27 09:15:00 | 338.20 | 2025-11-03 10:15:00 | 364.51 | TARGET | 26.31 |
| BUY | 2025-10-15 10:15:00 | 338.95 | 2025-11-04 09:15:00 | 371.74 | TARGET | 32.79 |
