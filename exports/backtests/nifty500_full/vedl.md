# Vedanta Ltd. (VEDL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 271.55
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 5 |
| ENTRY1 | 6 |
| ENTRY2 | 3 |
| EXIT | 6 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 5 / 4
- **Target hits / EMA400 exits:** 5 / 4
- **Total realized P&L (per unit):** 46.65
- **Avg P&L per closed trade:** 5.18

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 11:15:00 | 248.90 | 235.64 | 235.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 09:15:00 | 251.50 | 236.30 | 235.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 10:15:00 | 259.80 | 260.32 | 252.98 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-24 14:15:00 | 262.50 | 260.03 | 253.22 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 13:15:00 | 263.00 | 268.38 | 262.87 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-02-28 09:15:00 | 269.40 | 268.31 | 262.92 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-02-28 12:15:00 | 262.30 | 268.19 | 262.94 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 13:15:00 | 450.75 | 463.75 | 463.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 14:15:00 | 447.55 | 463.59 | 463.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 459.15 | 456.76 | 459.69 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 11:15:00 | 489.75 | 462.06 | 461.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 12:15:00 | 496.70 | 462.40 | 462.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 476.85 | 482.21 | 473.92 | EMA200 retest candle locked |

### Cycle 4 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 445.45 | 468.18 | 468.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 444.90 | 467.95 | 468.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-17 09:15:00 | 455.15 | 454.37 | 460.31 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-22 09:15:00 | 448.00 | 454.84 | 459.96 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-07 10:15:00 | 454.45 | 444.66 | 452.01 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 11:15:00 | 463.25 | 442.73 | 442.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-26 09:15:00 | 470.25 | 443.76 | 443.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 09:15:00 | 441.45 | 449.07 | 446.21 | EMA200 retest candle locked |

### Cycle 6 — SELL (started 2025-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 11:15:00 | 372.95 | 443.61 | 443.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 12:15:00 | 367.90 | 442.86 | 443.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 09:15:00 | 426.00 | 420.91 | 429.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-25 10:15:00 | 413.00 | 421.06 | 429.64 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-12 09:15:00 | 425.35 | 417.85 | 425.31 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 12:15:00 | 450.55 | 430.38 | 430.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 453.85 | 432.60 | 431.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 09:15:00 | 429.95 | 433.39 | 431.93 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-02 12:15:00 | 433.90 | 433.33 | 431.92 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 433.90 | 433.33 | 431.92 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-03 09:15:00 | 434.30 | 433.31 | 431.93 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 434.10 | 433.36 | 431.98 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-04 09:15:00 | 434.20 | 433.37 | 432.01 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 441.40 | 445.90 | 439.84 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-19 14:15:00 | 439.75 | 445.84 | 439.84 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 15:15:00 | 430.95 | 443.77 | 443.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 10:15:00 | 430.50 | 441.71 | 442.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 443.30 | 440.65 | 442.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-14 09:15:00 | 431.05 | 440.49 | 441.98 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-19 09:15:00 | 448.05 | 439.76 | 441.50 | Close above EMA400 |

### Cycle 9 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 462.10 | 441.29 | 441.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 12:15:00 | 468.55 | 441.78 | 441.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 446.05 | 447.13 | 444.56 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-29 09:15:00 | 453.90 | 447.21 | 444.64 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 665.85 | 693.10 | 665.56 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-19 14:15:00 | 665.30 | 692.82 | 665.56 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-24 14:15:00 | 262.50 | 2024-02-28 12:15:00 | 262.30 | EXIT_EMA400 | -0.20 |
| BUY | 2024-02-28 09:15:00 | 269.40 | 2024-02-28 12:15:00 | 262.30 | EXIT_EMA400 | -7.10 |
| SELL | 2025-01-22 09:15:00 | 448.00 | 2025-02-03 09:15:00 | 412.11 | TARGET | 35.89 |
| SELL | 2025-04-25 10:15:00 | 413.00 | 2025-05-12 09:15:00 | 425.35 | EXIT_EMA400 | -12.35 |
| BUY | 2025-06-02 12:15:00 | 433.90 | 2025-06-03 09:15:00 | 439.85 | TARGET | 5.95 |
| BUY | 2025-06-03 09:15:00 | 434.30 | 2025-06-05 14:15:00 | 441.40 | TARGET | 7.10 |
| BUY | 2025-06-04 09:15:00 | 434.20 | 2025-06-05 14:15:00 | 440.77 | TARGET | 6.57 |
| SELL | 2025-08-14 09:15:00 | 431.05 | 2025-08-19 09:15:00 | 448.05 | EXIT_EMA400 | -17.00 |
| BUY | 2025-09-29 09:15:00 | 453.90 | 2025-10-09 11:15:00 | 481.69 | TARGET | 27.79 |
