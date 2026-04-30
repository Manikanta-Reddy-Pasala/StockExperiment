# Tata Power Co. Ltd. (TATAPOWER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 444.05
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 6 |
| ENTRY1 | 6 |
| ENTRY2 | 3 |
| EXIT | 6 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 4 / 5
- **Target hits / EMA400 exits:** 4 / 5
- **Total realized P&L (per unit):** 25.36
- **Avg P&L per closed trade:** 2.82

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 13:15:00 | 413.10 | 430.18 | 430.24 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-09-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 12:15:00 | 443.15 | 429.09 | 429.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 09:15:00 | 446.55 | 429.65 | 429.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 10:15:00 | 447.35 | 451.84 | 442.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-08 10:15:00 | 459.20 | 451.50 | 442.74 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 452.25 | 455.41 | 447.12 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-18 11:15:00 | 454.15 | 455.37 | 447.18 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 448.10 | 455.35 | 447.65 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-22 10:15:00 | 446.00 | 455.26 | 447.65 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 14:15:00 | 431.45 | 442.51 | 442.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 09:15:00 | 427.75 | 442.26 | 442.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 426.10 | 424.52 | 431.39 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-18 09:15:00 | 415.95 | 427.46 | 431.02 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-19 09:15:00 | 368.15 | 354.42 | 365.15 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 10:15:00 | 392.65 | 370.22 | 370.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 14:15:00 | 394.50 | 372.14 | 371.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 12:15:00 | 378.35 | 379.28 | 375.50 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-08 09:15:00 | 380.75 | 379.05 | 375.58 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-08 13:15:00 | 372.85 | 379.01 | 375.63 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 11:15:00 | 382.80 | 395.15 | 395.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 380.95 | 391.14 | 392.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 11:15:00 | 388.10 | 387.64 | 390.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-04 12:15:00 | 384.85 | 387.58 | 390.46 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 388.60 | 386.82 | 389.71 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-10 11:15:00 | 390.25 | 386.87 | 389.71 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 399.60 | 390.68 | 390.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 401.45 | 392.04 | 391.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 395.65 | 397.50 | 394.67 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 12:15:00 | 384.00 | 392.99 | 393.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 15:15:00 | 382.00 | 392.69 | 392.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 09:15:00 | 394.70 | 391.79 | 392.38 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-28 12:15:00 | 391.30 | 391.84 | 392.38 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 391.30 | 391.84 | 392.38 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-28 13:15:00 | 390.35 | 391.82 | 392.37 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 388.25 | 391.75 | 392.32 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-01 11:15:00 | 387.10 | 391.66 | 392.27 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-01-02 10:15:00 | 388.80 | 382.26 | 385.60 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 405.10 | 375.14 | 375.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 09:15:00 | 415.00 | 382.67 | 379.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 15:15:00 | 382.90 | 384.22 | 380.23 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-24 09:15:00 | 388.40 | 384.26 | 380.27 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 384.85 | 385.06 | 381.09 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-30 14:15:00 | 378.40 | 384.92 | 381.12 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-10-08 10:15:00 | 459.20 | 2024-10-22 10:15:00 | 446.00 | EXIT_EMA400 | -13.20 |
| BUY | 2024-10-18 11:15:00 | 454.15 | 2024-10-22 10:15:00 | 446.00 | EXIT_EMA400 | -8.15 |
| SELL | 2024-12-18 09:15:00 | 415.95 | 2025-01-09 09:15:00 | 370.75 | TARGET | 45.20 |
| BUY | 2025-05-08 09:15:00 | 380.75 | 2025-05-08 13:15:00 | 372.85 | EXIT_EMA400 | -7.90 |
| SELL | 2025-09-04 12:15:00 | 384.85 | 2025-09-10 11:15:00 | 390.25 | EXIT_EMA400 | -5.40 |
| SELL | 2025-11-28 12:15:00 | 391.30 | 2025-12-01 09:15:00 | 388.07 | TARGET | 3.23 |
| SELL | 2025-11-28 13:15:00 | 390.35 | 2025-12-03 09:15:00 | 384.30 | TARGET | 6.05 |
| SELL | 2025-12-01 11:15:00 | 387.10 | 2025-12-09 09:15:00 | 371.58 | TARGET | 15.52 |
| BUY | 2026-03-24 09:15:00 | 388.40 | 2026-03-30 14:15:00 | 378.40 | EXIT_EMA400 | -10.00 |
