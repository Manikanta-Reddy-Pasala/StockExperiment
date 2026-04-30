# ITC Ltd. (ITC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 314.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 6 |
| ENTRY1 | 8 |
| ENTRY2 | 4 |
| EXIT | 8 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 3 / 9
- **Target hits / EMA400 exits:** 3 / 9
- **Total realized P&L (per unit):** -41.02
- **Avg P&L per closed trade:** -3.42

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 15:15:00 | 458.20 | 443.20 | 443.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 09:15:00 | 458.95 | 443.36 | 443.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 461.55 | 461.66 | 455.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-19 09:15:00 | 469.55 | 461.91 | 456.18 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 14:15:00 | 459.90 | 462.63 | 456.88 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-01-24 10:15:00 | 465.70 | 462.62 | 456.96 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-01-25 10:15:00 | 455.40 | 462.48 | 457.08 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-02-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 12:15:00 | 431.90 | 453.22 | 453.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 09:15:00 | 424.45 | 452.31 | 452.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-13 09:15:00 | 430.60 | 417.65 | 428.66 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-13 12:15:00 | 423.80 | 417.94 | 428.64 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 12:15:00 | 428.05 | 418.56 | 428.23 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-03-15 14:15:00 | 418.50 | 418.64 | 428.18 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-03-22 09:15:00 | 426.60 | 418.07 | 426.53 | Close above EMA400 |

### Cycle 3 — BUY (started 2024-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 13:15:00 | 439.40 | 428.47 | 428.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-07 09:15:00 | 445.15 | 429.70 | 429.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 12:15:00 | 427.80 | 431.22 | 429.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-10 09:15:00 | 435.25 | 431.11 | 429.92 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 435.25 | 431.11 | 429.92 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-05-14 09:15:00 | 428.95 | 431.25 | 430.08 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 12:15:00 | 420.45 | 430.49 | 430.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 14:15:00 | 419.50 | 430.28 | 430.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 14:15:00 | 429.70 | 428.27 | 429.26 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-07-02 13:15:00 | 424.70 | 428.24 | 429.21 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 429.00 | 428.19 | 429.17 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-07-04 10:15:00 | 430.55 | 428.20 | 429.14 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 11:15:00 | 453.10 | 429.98 | 429.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 10:15:00 | 456.20 | 432.64 | 431.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 12:15:00 | 504.00 | 509.98 | 497.02 | EMA200 retest candle locked |

### Cycle 6 — SELL (started 2024-11-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 13:15:00 | 479.65 | 492.62 | 492.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 09:15:00 | 476.75 | 491.46 | 492.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 480.95 | 479.79 | 484.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-28 10:15:00 | 475.40 | 479.75 | 484.67 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 472.65 | 471.84 | 477.46 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-23 13:15:00 | 471.85 | 471.84 | 477.43 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-12-24 10:15:00 | 478.75 | 471.98 | 477.39 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 434.50 | 423.92 | 423.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 435.20 | 424.04 | 423.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 09:15:00 | 425.95 | 427.63 | 425.98 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-23 09:15:00 | 434.90 | 427.56 | 426.00 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-28 09:15:00 | 421.65 | 429.36 | 427.11 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 418.00 | 425.26 | 425.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 417.15 | 424.63 | 424.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 11:15:00 | 419.90 | 418.58 | 420.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-10 10:15:00 | 417.25 | 418.59 | 420.82 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-07-15 09:15:00 | 420.60 | 418.47 | 420.54 | Close above EMA400 |

### Cycle 9 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 425.15 | 410.07 | 410.05 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 14:15:00 | 405.30 | 410.15 | 410.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 403.95 | 409.36 | 409.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 409.40 | 409.21 | 409.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-17 10:15:00 | 407.10 | 409.19 | 409.63 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 407.10 | 409.19 | 409.63 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-17 12:15:00 | 406.80 | 409.15 | 409.61 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-12-23 14:15:00 | 407.10 | 403.72 | 405.56 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-19 09:15:00 | 469.55 | 2024-01-25 10:15:00 | 455.40 | EXIT_EMA400 | -14.15 |
| BUY | 2024-01-24 10:15:00 | 465.70 | 2024-01-25 10:15:00 | 455.40 | EXIT_EMA400 | -10.30 |
| SELL | 2024-03-13 12:15:00 | 423.80 | 2024-03-19 11:15:00 | 409.29 | TARGET | 14.51 |
| SELL | 2024-03-15 14:15:00 | 418.50 | 2024-03-22 09:15:00 | 426.60 | EXIT_EMA400 | -8.10 |
| BUY | 2024-05-10 09:15:00 | 435.25 | 2024-05-14 09:15:00 | 428.95 | EXIT_EMA400 | -6.30 |
| SELL | 2024-07-02 13:15:00 | 424.70 | 2024-07-04 10:15:00 | 430.55 | EXIT_EMA400 | -5.85 |
| SELL | 2024-11-28 10:15:00 | 475.40 | 2024-12-24 10:15:00 | 478.75 | EXIT_EMA400 | -3.35 |
| SELL | 2024-12-23 13:15:00 | 471.85 | 2024-12-24 10:15:00 | 478.75 | EXIT_EMA400 | -6.90 |
| BUY | 2025-05-23 09:15:00 | 434.90 | 2025-05-28 09:15:00 | 421.65 | EXIT_EMA400 | -13.25 |
| SELL | 2025-07-10 10:15:00 | 417.25 | 2025-07-15 09:15:00 | 420.60 | EXIT_EMA400 | -3.35 |
| SELL | 2025-11-17 10:15:00 | 407.10 | 2025-12-01 09:15:00 | 399.50 | TARGET | 7.60 |
| SELL | 2025-11-17 12:15:00 | 406.80 | 2025-12-01 09:15:00 | 398.38 | TARGET | 8.42 |
