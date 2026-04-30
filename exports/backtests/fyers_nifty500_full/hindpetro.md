# Hindustan Petroleum Corporation Ltd. (HINDPETRO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 374.65
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 68.63
- **Avg P&L per closed trade:** 11.44

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 11:15:00 | 391.80 | 397.23 | 397.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 386.05 | 397.05 | 397.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 09:15:00 | 385.55 | 384.38 | 389.81 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-27 10:15:00 | 377.80 | 384.14 | 389.47 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-04 09:15:00 | 389.35 | 383.56 | 388.31 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 14:15:00 | 415.10 | 391.62 | 391.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 10:15:00 | 417.60 | 396.82 | 394.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 399.55 | 402.63 | 398.38 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 13:15:00 | 371.00 | 395.16 | 395.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 12:15:00 | 366.80 | 393.79 | 394.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 10:15:00 | 333.00 | 329.56 | 348.45 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-05 13:15:00 | 325.55 | 329.53 | 348.15 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-21 11:15:00 | 342.50 | 329.58 | 342.36 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 13:15:00 | 379.80 | 350.02 | 349.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 11:15:00 | 383.60 | 353.43 | 351.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 396.85 | 401.81 | 389.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-26 09:15:00 | 412.65 | 398.80 | 391.41 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-31 09:15:00 | 411.50 | 426.94 | 416.29 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 10:15:00 | 394.70 | 410.55 | 410.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 393.80 | 410.22 | 410.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 398.30 | 396.35 | 401.68 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 14:15:00 | 423.05 | 404.64 | 404.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 09:15:00 | 435.70 | 406.34 | 405.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 13:15:00 | 465.00 | 467.20 | 450.87 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-19 14:15:00 | 469.75 | 460.05 | 453.84 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 464.90 | 472.85 | 463.57 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-08 10:15:00 | 458.50 | 472.71 | 463.54 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 421.20 | 457.24 | 457.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 418.25 | 456.85 | 457.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 447.70 | 447.14 | 451.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-19 15:15:00 | 433.30 | 451.47 | 452.86 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 439.00 | 448.71 | 451.24 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-25 11:15:00 | 434.15 | 448.46 | 451.10 | Sell entry 2 (retest2 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-27 10:15:00 | 377.80 | 2024-12-04 09:15:00 | 389.35 | EXIT_EMA400 | -11.55 |
| SELL | 2025-03-05 13:15:00 | 325.55 | 2025-03-21 11:15:00 | 342.50 | EXIT_EMA400 | -16.95 |
| BUY | 2025-06-26 09:15:00 | 412.65 | 2025-07-31 09:15:00 | 411.50 | EXIT_EMA400 | -1.15 |
| BUY | 2025-12-19 14:15:00 | 469.75 | 2026-01-08 10:15:00 | 458.50 | EXIT_EMA400 | -11.25 |
| SELL | 2026-02-19 15:15:00 | 433.30 | 2026-03-09 09:15:00 | 374.61 | TARGET | 58.69 |
| SELL | 2026-02-25 11:15:00 | 434.15 | 2026-03-09 09:15:00 | 383.31 | TARGET | 50.84 |
