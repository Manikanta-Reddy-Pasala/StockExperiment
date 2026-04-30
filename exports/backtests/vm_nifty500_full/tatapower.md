# Tata Power Co. Ltd. (TATAPOWER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 444.55
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** 43.86
- **Avg P&L per closed trade:** 6.27

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 10:15:00 | 408.55 | 430.94 | 430.98 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 14:15:00 | 441.40 | 429.38 | 429.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 09:15:00 | 446.50 | 429.68 | 429.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 10:15:00 | 447.35 | 451.87 | 442.73 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-08 10:15:00 | 459.20 | 451.52 | 442.87 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 452.30 | 455.42 | 447.22 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-18 11:15:00 | 454.15 | 455.38 | 447.28 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 448.10 | 455.36 | 447.74 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-22 10:15:00 | 445.90 | 455.27 | 447.74 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 13:15:00 | 433.00 | 442.61 | 442.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 09:15:00 | 427.75 | 442.25 | 442.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 426.00 | 424.52 | 431.41 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-18 09:15:00 | 415.95 | 427.46 | 431.03 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-19 09:15:00 | 368.00 | 354.43 | 365.38 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 11:15:00 | 391.85 | 370.43 | 370.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 14:15:00 | 394.30 | 372.13 | 371.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 12:15:00 | 378.35 | 379.27 | 375.57 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-08 09:15:00 | 380.95 | 379.05 | 375.65 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-08 13:15:00 | 373.10 | 379.01 | 375.70 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 11:15:00 | 382.80 | 395.15 | 395.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 381.00 | 391.15 | 392.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 11:15:00 | 388.10 | 387.65 | 390.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-04 12:15:00 | 384.85 | 387.59 | 390.47 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 388.50 | 386.82 | 389.72 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-10 11:15:00 | 390.25 | 386.87 | 389.71 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 399.65 | 390.68 | 390.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 401.30 | 392.05 | 391.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 395.80 | 397.50 | 394.67 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 12:15:00 | 384.00 | 392.97 | 393.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 10:15:00 | 381.95 | 392.47 | 392.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 09:15:00 | 394.70 | 391.78 | 392.38 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-27 12:15:00 | 392.15 | 391.82 | 392.39 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 392.15 | 391.82 | 392.39 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-28 09:15:00 | 392.60 | 391.83 | 392.38 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 13:15:00 | 404.85 | 375.50 | 375.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 09:15:00 | 415.10 | 382.71 | 379.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 09:15:00 | 384.85 | 385.13 | 381.24 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-08 09:15:00 | 392.70 | 384.33 | 381.41 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-10-08 10:15:00 | 459.20 | 2024-10-22 10:15:00 | 445.90 | EXIT_EMA400 | -13.30 |
| BUY | 2024-10-18 11:15:00 | 454.15 | 2024-10-22 10:15:00 | 445.90 | EXIT_EMA400 | -8.25 |
| SELL | 2024-12-18 09:15:00 | 415.95 | 2025-01-09 09:15:00 | 370.71 | TARGET | 45.24 |
| BUY | 2025-05-08 09:15:00 | 380.95 | 2025-05-08 13:15:00 | 373.10 | EXIT_EMA400 | -7.85 |
| SELL | 2025-09-04 12:15:00 | 384.85 | 2025-09-10 11:15:00 | 390.25 | EXIT_EMA400 | -5.40 |
| SELL | 2025-11-27 12:15:00 | 392.15 | 2025-11-28 09:15:00 | 392.60 | EXIT_EMA400 | -0.45 |
| BUY | 2026-04-08 09:15:00 | 392.70 | 2026-04-16 09:15:00 | 426.57 | TARGET | 33.87 |
