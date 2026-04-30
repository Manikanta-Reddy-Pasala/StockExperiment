# Birlasoft Ltd. (BSOFT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 369.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 28.34
- **Avg P&L per closed trade:** 4.05

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 14:15:00 | 587.55 | 655.25 | 655.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 09:15:00 | 583.80 | 653.87 | 654.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 10:15:00 | 633.25 | 631.48 | 641.50 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-19 10:15:00 | 614.10 | 643.23 | 645.22 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-26 11:15:00 | 593.10 | 572.75 | 590.05 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 10:15:00 | 434.45 | 420.57 | 420.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 11:15:00 | 436.55 | 421.63 | 421.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 426.25 | 426.96 | 424.23 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-15 12:15:00 | 428.75 | 426.01 | 424.02 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 425.05 | 426.97 | 424.71 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-18 12:15:00 | 422.35 | 426.91 | 424.70 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 398.75 | 422.67 | 422.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 396.80 | 422.17 | 422.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 11:15:00 | 420.10 | 417.05 | 419.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-30 12:15:00 | 413.40 | 417.01 | 419.69 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 413.40 | 417.01 | 419.69 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-07-30 13:15:00 | 412.45 | 416.97 | 419.65 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 387.75 | 381.28 | 390.94 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-19 10:15:00 | 382.80 | 381.71 | 390.78 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-23 09:15:00 | 380.00 | 358.90 | 371.46 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 14:15:00 | 388.10 | 376.52 | 376.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 399.70 | 379.36 | 378.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 11:15:00 | 422.60 | 424.15 | 409.93 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-07 09:15:00 | 431.45 | 424.12 | 410.27 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 414.00 | 425.60 | 412.45 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-12 10:15:00 | 409.65 | 425.44 | 412.44 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 12:15:00 | 379.15 | 414.33 | 414.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 13:15:00 | 378.40 | 413.97 | 414.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 14:15:00 | 370.50 | 368.73 | 383.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 09:15:00 | 361.55 | 369.05 | 381.52 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 13:15:00 | 380.95 | 369.26 | 380.95 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-16 09:15:00 | 390.40 | 369.68 | 380.99 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-19 10:15:00 | 614.10 | 2024-11-26 11:15:00 | 593.10 | EXIT_EMA400 | 21.00 |
| BUY | 2025-07-15 12:15:00 | 428.75 | 2025-07-18 12:15:00 | 422.35 | EXIT_EMA400 | -6.40 |
| SELL | 2025-07-30 12:15:00 | 413.40 | 2025-07-31 10:15:00 | 394.54 | TARGET | 18.86 |
| SELL | 2025-07-30 13:15:00 | 412.45 | 2025-07-31 15:15:00 | 390.84 | TARGET | 21.61 |
| SELL | 2025-09-19 10:15:00 | 382.80 | 2025-09-26 09:15:00 | 358.87 | TARGET | 23.93 |
| BUY | 2026-01-07 09:15:00 | 431.45 | 2026-01-12 10:15:00 | 409.65 | EXIT_EMA400 | -21.80 |
| SELL | 2026-04-13 09:15:00 | 361.55 | 2026-04-16 09:15:00 | 390.40 | EXIT_EMA400 | -28.85 |
