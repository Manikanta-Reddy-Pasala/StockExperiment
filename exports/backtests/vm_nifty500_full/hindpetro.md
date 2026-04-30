# Hindustan Petroleum Corporation Ltd. (HINDPETRO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 374.55
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 6 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -94.10
- **Avg P&L per closed trade:** -15.68

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-09 15:15:00 | 199.30 | 174.28 | 174.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-10 10:15:00 | 199.77 | 174.79 | 174.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 09:15:00 | 338.63 | 338.82 | 311.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-02 11:15:00 | 356.03 | 324.36 | 318.66 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-05-13 09:15:00 | 321.73 | 331.83 | 324.12 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 384.55 | 397.24 | 397.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 383.10 | 397.10 | 397.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 09:15:00 | 385.55 | 384.53 | 389.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-26 15:15:00 | 378.00 | 384.38 | 389.70 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-04 09:15:00 | 389.35 | 383.63 | 388.40 | Close above EMA400 |

### Cycle 3 — BUY (started 2024-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 15:15:00 | 414.55 | 391.89 | 391.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 10:15:00 | 417.60 | 396.83 | 394.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 399.55 | 402.64 | 398.42 | EMA200 retest candle locked |

### Cycle 4 — SELL (started 2025-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 13:15:00 | 371.30 | 395.18 | 395.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 12:15:00 | 366.80 | 393.79 | 394.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 10:15:00 | 333.00 | 329.95 | 349.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-05 13:15:00 | 325.70 | 329.90 | 348.70 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 11:15:00 | 342.65 | 329.77 | 342.75 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-21 12:15:00 | 348.35 | 329.95 | 342.77 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-04-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 14:15:00 | 382.40 | 350.41 | 350.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 11:15:00 | 383.60 | 353.49 | 351.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 396.90 | 401.82 | 389.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 09:15:00 | 408.10 | 397.54 | 390.32 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-31 09:15:00 | 411.50 | 426.93 | 416.30 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 10:15:00 | 394.70 | 410.55 | 410.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 11:15:00 | 394.10 | 410.39 | 410.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 398.25 | 396.35 | 401.70 | EMA200 retest candle locked |

### Cycle 7 — BUY (started 2025-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 14:15:00 | 423.05 | 404.64 | 404.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 09:15:00 | 435.70 | 406.34 | 405.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 13:15:00 | 465.00 | 467.24 | 450.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-19 14:15:00 | 469.70 | 460.07 | 453.87 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 464.90 | 472.86 | 463.59 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-08 10:15:00 | 458.60 | 472.72 | 463.56 | Close below EMA400 |

### Cycle 8 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 423.80 | 457.63 | 457.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 421.20 | 457.26 | 457.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 452.60 | 448.73 | 452.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-03 10:15:00 | 445.15 | 448.74 | 452.61 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-04 09:15:00 | 463.25 | 448.90 | 452.58 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-05-02 11:15:00 | 356.03 | 2024-05-13 09:15:00 | 321.73 | EXIT_EMA400 | -34.30 |
| SELL | 2024-11-26 15:15:00 | 378.00 | 2024-12-04 09:15:00 | 389.35 | EXIT_EMA400 | -11.35 |
| SELL | 2025-03-05 13:15:00 | 325.70 | 2025-03-21 12:15:00 | 348.35 | EXIT_EMA400 | -22.65 |
| BUY | 2025-06-24 09:15:00 | 408.10 | 2025-07-31 09:15:00 | 411.50 | EXIT_EMA400 | 3.40 |
| BUY | 2025-12-19 14:15:00 | 469.70 | 2026-01-08 10:15:00 | 458.60 | EXIT_EMA400 | -11.10 |
| SELL | 2026-02-03 10:15:00 | 445.15 | 2026-02-04 09:15:00 | 463.25 | EXIT_EMA400 | -18.10 |
