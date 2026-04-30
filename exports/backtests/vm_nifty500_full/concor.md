# Container Corporation of India Ltd. (CONCOR.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 508.85
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 55.43
- **Avg P&L per closed trade:** 13.86

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 09:15:00 | 528.76 | 538.99 | 539.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-28 09:15:00 | 525.88 | 538.26 | 538.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-30 09:15:00 | 537.60 | 537.54 | 538.23 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 15:15:00 | 551.04 | 538.97 | 538.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 09:15:00 | 553.56 | 539.12 | 538.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 12:15:00 | 561.40 | 566.37 | 556.83 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-10-04 14:15:00 | 570.40 | 566.41 | 556.94 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 13:15:00 | 559.28 | 566.67 | 558.59 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2023-10-12 09:15:00 | 557.76 | 566.43 | 558.59 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 14:15:00 | 796.00 | 825.31 | 825.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 09:15:00 | 788.16 | 824.73 | 825.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 10:15:00 | 662.76 | 660.93 | 690.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-12 12:15:00 | 655.24 | 668.38 | 686.25 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 10:15:00 | 564.28 | 541.77 | 567.49 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-24 12:15:00 | 569.92 | 542.28 | 567.49 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 585.24 | 561.68 | 561.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 589.92 | 561.96 | 561.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 604.40 | 604.93 | 588.51 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 10:15:00 | 611.72 | 605.00 | 588.63 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-19 10:15:00 | 589.36 | 604.57 | 590.06 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 537.50 | 595.01 | 595.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 534.00 | 594.40 | 594.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 561.85 | 553.65 | 565.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-23 09:15:00 | 545.40 | 555.37 | 564.43 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-29 11:15:00 | 548.00 | 538.44 | 547.18 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 15:15:00 | 513.55 | 486.85 | 486.81 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-10-04 14:15:00 | 570.40 | 2023-10-12 09:15:00 | 557.76 | EXIT_EMA400 | -12.64 |
| SELL | 2024-12-12 12:15:00 | 655.24 | 2025-02-11 09:15:00 | 562.21 | TARGET | 93.03 |
| BUY | 2025-06-16 10:15:00 | 611.72 | 2025-06-19 10:15:00 | 589.36 | EXIT_EMA400 | -22.36 |
| SELL | 2025-09-23 09:15:00 | 545.40 | 2025-10-29 11:15:00 | 548.00 | EXIT_EMA400 | -2.60 |
