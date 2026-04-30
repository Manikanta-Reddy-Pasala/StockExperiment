# Leela Palaces Hotels & Resorts Ltd. (THELEELA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2025-06-02 09:15:00 → 2026-04-30 15:30:00 (1574 bars)
- **Last close:** 428.25
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -57.75
- **Avg P&L per closed trade:** -14.44

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 15:15:00 | 395.10 | 425.60 | 425.68 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 13:15:00 | 426.15 | 423.09 | 423.08 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 422.35 | 423.07 | 423.07 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 13:15:00 | 428.95 | 423.12 | 423.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-08 14:15:00 | 430.40 | 423.20 | 423.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 09:15:00 | 432.30 | 433.41 | 429.23 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-30 11:15:00 | 439.55 | 432.67 | 429.41 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-06 09:15:00 | 422.95 | 433.78 | 430.39 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 13:15:00 | 409.60 | 428.86 | 428.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 408.00 | 426.24 | 427.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 14:15:00 | 414.50 | 410.74 | 417.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-18 09:15:00 | 408.25 | 410.75 | 417.55 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 416.75 | 410.81 | 417.34 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-23 14:15:00 | 418.60 | 411.22 | 416.96 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 15:15:00 | 425.10 | 420.93 | 420.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 09:15:00 | 439.00 | 421.52 | 421.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 413.25 | 426.17 | 423.75 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 11:15:00 | 407.10 | 421.66 | 421.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 15:15:00 | 405.00 | 421.14 | 421.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 13:15:00 | 420.20 | 418.66 | 420.09 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 10:15:00 | 447.00 | 421.44 | 421.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 14:15:00 | 453.00 | 427.60 | 424.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-05 14:15:00 | 433.65 | 435.59 | 430.96 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-09 15:15:00 | 440.00 | 434.86 | 430.91 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 440.00 | 434.86 | 430.91 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-10 10:15:00 | 430.20 | 434.82 | 430.94 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 09:15:00 | 404.55 | 428.37 | 428.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 10:15:00 | 403.25 | 428.12 | 428.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 10:15:00 | 420.00 | 419.10 | 422.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-07 14:15:00 | 414.00 | 419.06 | 422.75 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-08 09:15:00 | 435.00 | 419.17 | 422.77 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 10:15:00 | 430.55 | 425.15 | 425.14 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 409.60 | 425.09 | 425.12 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 433.00 | 425.17 | 425.16 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 11:15:00 | 418.40 | 425.09 | 425.12 | EMA200 below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-10-30 11:15:00 | 439.55 | 2025-11-06 09:15:00 | 422.95 | EXIT_EMA400 | -16.60 |
| SELL | 2025-12-18 09:15:00 | 408.25 | 2025-12-23 14:15:00 | 418.60 | EXIT_EMA400 | -10.35 |
| BUY | 2026-03-09 15:15:00 | 440.00 | 2026-03-10 10:15:00 | 430.20 | EXIT_EMA400 | -9.80 |
| SELL | 2026-04-07 14:15:00 | 414.00 | 2026-04-08 09:15:00 | 435.00 | EXIT_EMA400 | -21.00 |
