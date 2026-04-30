# Delhivery Ltd. (DELHIVERY.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 467.05
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 4 |
| ENTRY1 | 7 |
| ENTRY2 | 2 |
| EXIT | 7 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 5 / 4
- **Target hits / EMA400 exits:** 4 / 5
- **Total realized P&L (per unit):** 194.14
- **Avg P&L per closed trade:** 21.57

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 09:15:00 | 407.50 | 415.64 | 415.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 12:15:00 | 405.55 | 414.90 | 415.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-16 09:15:00 | 414.95 | 414.15 | 414.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-17 09:15:00 | 403.10 | 414.01 | 414.78 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 09:15:00 | 393.45 | 386.32 | 395.80 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-12-26 13:15:00 | 388.40 | 386.52 | 395.72 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-01-03 09:15:00 | 401.50 | 386.61 | 394.16 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 12:15:00 | 430.70 | 398.65 | 398.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 09:15:00 | 440.25 | 401.94 | 400.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 11:15:00 | 424.25 | 426.29 | 414.74 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-12 13:15:00 | 432.40 | 426.38 | 414.90 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 15:15:00 | 420.00 | 426.26 | 414.95 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-02-13 09:15:00 | 425.65 | 426.25 | 415.00 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 438.50 | 452.15 | 438.00 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-03-13 10:15:00 | 425.30 | 451.88 | 437.94 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 12:15:00 | 392.60 | 445.84 | 446.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 13:15:00 | 386.65 | 428.98 | 436.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 10:15:00 | 395.95 | 389.96 | 401.44 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 09:15:00 | 431.10 | 406.70 | 406.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 09:15:00 | 439.40 | 420.18 | 415.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 13:15:00 | 421.60 | 422.83 | 417.98 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 13:15:00 | 397.35 | 415.11 | 415.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 14:15:00 | 396.30 | 414.92 | 415.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 11:15:00 | 356.50 | 351.25 | 368.88 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-13 09:15:00 | 327.50 | 350.68 | 360.33 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-17 09:15:00 | 276.35 | 259.92 | 276.27 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 12:15:00 | 301.95 | 286.02 | 285.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 306.70 | 287.62 | 286.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 11:15:00 | 350.00 | 350.43 | 332.15 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 12:15:00 | 354.15 | 350.47 | 332.26 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-26 14:15:00 | 444.65 | 463.89 | 446.37 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 10:15:00 | 430.35 | 451.14 | 451.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 12:15:00 | 428.30 | 450.72 | 450.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 14:15:00 | 413.10 | 411.53 | 422.05 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-09 13:15:00 | 404.60 | 412.38 | 421.18 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-30 09:15:00 | 415.45 | 403.35 | 412.50 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 15:15:00 | 429.70 | 419.22 | 419.22 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 415.40 | 419.18 | 419.20 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 12:15:00 | 424.55 | 419.24 | 419.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 427.80 | 419.34 | 419.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 12:15:00 | 426.60 | 426.83 | 423.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-02 13:15:00 | 428.30 | 426.85 | 423.62 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 428.30 | 426.85 | 423.62 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-04 09:15:00 | 419.55 | 426.79 | 423.64 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-03-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 11:15:00 | 398.70 | 421.36 | 421.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 12:15:00 | 394.65 | 421.09 | 421.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 12:15:00 | 421.25 | 419.26 | 420.35 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-19 09:15:00 | 414.20 | 419.31 | 420.35 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-03-20 09:15:00 | 423.65 | 419.01 | 420.16 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 431.00 | 420.83 | 420.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 09:15:00 | 445.55 | 421.36 | 421.09 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-17 09:15:00 | 403.10 | 2023-12-13 09:15:00 | 368.06 | TARGET | 35.04 |
| SELL | 2023-12-26 13:15:00 | 388.40 | 2024-01-03 09:15:00 | 401.50 | EXIT_EMA400 | -13.10 |
| BUY | 2024-02-13 09:15:00 | 425.65 | 2024-02-15 10:15:00 | 457.59 | TARGET | 31.94 |
| BUY | 2024-02-12 13:15:00 | 432.40 | 2024-02-27 11:15:00 | 484.91 | TARGET | 52.51 |
| SELL | 2025-01-13 09:15:00 | 327.50 | 2025-04-17 09:15:00 | 276.35 | EXIT_EMA400 | 51.15 |
| BUY | 2025-06-20 12:15:00 | 354.15 | 2025-07-09 12:15:00 | 419.81 | TARGET | 65.66 |
| SELL | 2026-01-09 13:15:00 | 404.60 | 2026-01-30 09:15:00 | 415.45 | EXIT_EMA400 | -10.85 |
| BUY | 2026-03-02 13:15:00 | 428.30 | 2026-03-04 09:15:00 | 419.55 | EXIT_EMA400 | -8.75 |
| SELL | 2026-03-19 09:15:00 | 414.20 | 2026-03-20 09:15:00 | 423.65 | EXIT_EMA400 | -9.45 |
