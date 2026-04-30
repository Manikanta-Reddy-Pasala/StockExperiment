# Jupiter Wagons Ltd. (JWL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 283.00
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
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** 130.47
- **Avg P&L per closed trade:** 18.64

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 11:15:00 | 551.50 | 605.07 | 605.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-16 12:15:00 | 549.60 | 604.51 | 604.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 13:15:00 | 561.50 | 558.04 | 574.57 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-16 09:15:00 | 546.25 | 557.95 | 574.28 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-01 17:15:00 | 526.00 | 503.46 | 524.56 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 09:15:00 | 530.00 | 508.23 | 508.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-20 14:15:00 | 543.55 | 509.24 | 508.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 14:15:00 | 510.65 | 511.66 | 510.01 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-03 09:15:00 | 521.00 | 509.01 | 508.83 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 521.00 | 509.01 | 508.83 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-06 09:15:00 | 499.00 | 509.15 | 508.91 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 482.85 | 508.65 | 508.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 12:15:00 | 473.00 | 503.41 | 505.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-16 11:15:00 | 487.45 | 486.33 | 496.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-22 10:15:00 | 466.30 | 487.13 | 495.45 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-24 09:15:00 | 365.90 | 324.38 | 362.60 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 12:15:00 | 445.95 | 369.53 | 369.35 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 14:15:00 | 369.30 | 382.23 | 382.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-15 09:15:00 | 368.30 | 381.96 | 382.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 368.95 | 345.02 | 357.83 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-21 14:15:00 | 344.90 | 345.64 | 357.82 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 344.90 | 345.64 | 357.82 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-21 15:15:00 | 341.40 | 345.59 | 357.74 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-15 09:15:00 | 346.00 | 333.83 | 345.38 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 325.40 | 309.06 | 308.98 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 14:15:00 | 293.00 | 308.94 | 308.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 290.30 | 308.62 | 308.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 11:15:00 | 326.20 | 306.87 | 307.86 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 09:15:00 | 326.40 | 308.96 | 308.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 334.45 | 312.50 | 310.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 13:15:00 | 310.30 | 312.77 | 311.02 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 299.80 | 309.67 | 309.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 293.80 | 309.02 | 309.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 10:15:00 | 287.95 | 287.74 | 296.73 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-10 09:15:00 | 280.55 | 288.12 | 296.37 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 276.95 | 268.41 | 280.38 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-07 13:15:00 | 271.68 | 268.49 | 280.30 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 275.16 | 268.50 | 278.63 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-16 10:15:00 | 282.80 | 268.94 | 278.45 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-16 09:15:00 | 546.25 | 2024-10-07 10:15:00 | 462.16 | TARGET | 84.09 |
| BUY | 2025-01-03 09:15:00 | 521.00 | 2025-01-06 09:15:00 | 499.00 | EXIT_EMA400 | -22.00 |
| SELL | 2025-01-22 10:15:00 | 466.30 | 2025-01-28 09:15:00 | 378.85 | TARGET | 87.45 |
| SELL | 2025-08-21 14:15:00 | 344.90 | 2025-09-15 09:15:00 | 346.00 | EXIT_EMA400 | -1.10 |
| SELL | 2025-08-21 15:15:00 | 341.40 | 2025-09-15 09:15:00 | 346.00 | EXIT_EMA400 | -4.60 |
| SELL | 2026-03-10 09:15:00 | 280.55 | 2026-04-16 10:15:00 | 282.80 | EXIT_EMA400 | -2.25 |
| SELL | 2026-04-07 13:15:00 | 271.68 | 2026-04-16 10:15:00 | 282.80 | EXIT_EMA400 | -11.12 |
