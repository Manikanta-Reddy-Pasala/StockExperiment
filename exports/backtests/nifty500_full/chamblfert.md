# Chambal Fertilizers & Chemicals Ltd. (CHAMBLFERT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 438.05
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
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / EMA400 exits:** 4 / 2
- **Total realized P&L (per unit):** 71.14
- **Avg P&L per closed trade:** 11.86

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 14:15:00 | 286.35 | 273.43 | 273.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-07 10:15:00 | 289.70 | 273.81 | 273.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 12:15:00 | 275.50 | 275.60 | 274.58 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-09-12 13:15:00 | 276.50 | 275.61 | 274.59 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 14:15:00 | 278.25 | 275.64 | 274.61 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-09-13 12:15:00 | 283.20 | 275.84 | 274.73 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 11:15:00 | 277.00 | 278.70 | 276.79 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2023-09-28 12:15:00 | 276.40 | 278.68 | 276.79 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 12:15:00 | 343.75 | 353.49 | 353.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-28 14:15:00 | 341.70 | 352.09 | 352.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 10:15:00 | 354.90 | 352.02 | 352.73 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2024-04-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-03 12:15:00 | 374.20 | 353.56 | 353.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-03 15:15:00 | 374.85 | 354.17 | 353.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 09:15:00 | 361.30 | 363.52 | 359.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-19 11:15:00 | 363.10 | 363.47 | 359.38 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 11:15:00 | 363.10 | 363.47 | 359.38 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-04-19 12:15:00 | 367.25 | 363.51 | 359.41 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 391.50 | 396.96 | 386.66 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-06-04 10:15:00 | 369.65 | 396.69 | 386.58 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 14:15:00 | 472.10 | 498.54 | 498.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 461.70 | 497.93 | 498.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 495.00 | 487.06 | 491.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 14:15:00 | 483.30 | 490.01 | 493.09 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-12 10:15:00 | 493.10 | 489.40 | 492.63 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 13:15:00 | 536.70 | 491.70 | 491.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 09:15:00 | 540.45 | 493.03 | 492.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 508.15 | 513.38 | 504.94 | EMA200 retest candle locked |

### Cycle 6 — SELL (started 2025-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 14:15:00 | 454.35 | 501.13 | 501.14 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 09:15:00 | 526.70 | 499.25 | 499.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 10:15:00 | 535.50 | 499.61 | 499.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 663.30 | 664.93 | 626.18 | EMA200 retest candle locked |

### Cycle 8 — SELL (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 11:15:00 | 551.20 | 610.35 | 610.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 12:15:00 | 549.55 | 609.74 | 610.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 10:15:00 | 558.95 | 558.34 | 570.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-24 12:15:00 | 554.60 | 558.35 | 570.37 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-05 12:15:00 | 565.35 | 547.62 | 561.36 | Close above EMA400 |

### Cycle 9 — BUY (started 2026-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 12:15:00 | 454.00 | 445.04 | 445.00 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-09-12 13:15:00 | 276.50 | 2023-09-13 09:15:00 | 282.24 | TARGET | 5.74 |
| BUY | 2023-09-13 12:15:00 | 283.20 | 2023-09-28 12:15:00 | 276.40 | EXIT_EMA400 | -6.80 |
| BUY | 2024-04-19 11:15:00 | 363.10 | 2024-04-22 14:15:00 | 374.27 | TARGET | 11.17 |
| BUY | 2024-04-19 12:15:00 | 367.25 | 2024-04-24 09:15:00 | 390.76 | TARGET | 23.51 |
| SELL | 2024-11-08 14:15:00 | 483.30 | 2024-11-12 10:15:00 | 493.10 | EXIT_EMA400 | -9.80 |
| SELL | 2025-07-24 12:15:00 | 554.60 | 2025-07-31 14:15:00 | 507.27 | TARGET | 47.32 |
