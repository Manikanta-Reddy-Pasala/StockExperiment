# Leela Palaces Hotels & Resorts Ltd. (THELEELA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2025-06-02 09:15:00 → 2026-04-30 15:15:00 (1584 bars)
- **Last close:** 423.35
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 0 / 6
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -92.35
- **Avg P&L per closed trade:** -15.39

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 15:15:00 | 395.10 | 425.62 | 425.69 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 15:15:00 | 425.90 | 423.07 | 423.07 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 09:15:00 | 419.25 | 423.03 | 423.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 15:15:00 | 418.15 | 422.89 | 422.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 425.55 | 422.91 | 422.99 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 13:15:00 | 428.95 | 423.06 | 423.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-08 14:15:00 | 430.40 | 423.13 | 423.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 09:15:00 | 432.30 | 433.37 | 429.20 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-30 11:15:00 | 439.55 | 432.60 | 429.36 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 437.45 | 433.15 | 429.83 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-11-04 10:15:00 | 443.15 | 433.50 | 430.14 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-11-06 09:15:00 | 422.95 | 433.67 | 430.33 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 13:15:00 | 409.60 | 428.78 | 428.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 408.00 | 426.20 | 427.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 14:15:00 | 413.90 | 410.75 | 417.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-17 15:15:00 | 408.35 | 410.79 | 417.59 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 408.70 | 410.77 | 417.54 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-23 14:15:00 | 418.60 | 411.22 | 416.95 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 14:15:00 | 430.80 | 420.88 | 420.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 09:15:00 | 439.00 | 421.51 | 421.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 413.25 | 426.12 | 423.72 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 11:15:00 | 407.10 | 421.64 | 421.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 15:15:00 | 405.00 | 421.12 | 421.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 13:15:00 | 420.20 | 418.64 | 420.07 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-02-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 15:15:00 | 446.10 | 421.36 | 421.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 14:15:00 | 453.00 | 427.87 | 424.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-05 14:15:00 | 433.65 | 435.69 | 431.05 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-09 15:15:00 | 440.00 | 434.95 | 431.00 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 440.00 | 434.95 | 431.00 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-10 10:15:00 | 430.20 | 434.91 | 431.02 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 09:15:00 | 404.55 | 428.32 | 428.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 10:15:00 | 403.25 | 428.07 | 428.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 10:15:00 | 420.00 | 419.00 | 422.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-07 14:15:00 | 414.00 | 418.98 | 422.72 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-08 09:15:00 | 435.00 | 419.09 | 422.73 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 15:15:00 | 435.00 | 425.14 | 425.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 09:15:00 | 435.55 | 425.24 | 425.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 13:15:00 | 424.40 | 425.41 | 425.25 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-29 10:15:00 | 433.00 | 425.29 | 425.21 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 429.45 | 425.38 | 425.26 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-04-30 09:15:00 | 418.50 | 425.33 | 425.24 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 13:15:00 | 420.55 | 425.10 | 425.12 | EMA200 below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-10-30 11:15:00 | 439.55 | 2025-11-06 09:15:00 | 422.95 | EXIT_EMA400 | -16.60 |
| BUY | 2025-11-04 10:15:00 | 443.15 | 2025-11-06 09:15:00 | 422.95 | EXIT_EMA400 | -20.20 |
| SELL | 2025-12-17 15:15:00 | 408.35 | 2025-12-23 14:15:00 | 418.60 | EXIT_EMA400 | -10.25 |
| BUY | 2026-03-09 15:15:00 | 440.00 | 2026-03-10 10:15:00 | 430.20 | EXIT_EMA400 | -9.80 |
| SELL | 2026-04-07 14:15:00 | 414.00 | 2026-04-08 09:15:00 | 435.00 | EXIT_EMA400 | -21.00 |
| BUY | 2026-04-29 10:15:00 | 433.00 | 2026-04-30 09:15:00 | 418.50 | EXIT_EMA400 | -14.50 |
