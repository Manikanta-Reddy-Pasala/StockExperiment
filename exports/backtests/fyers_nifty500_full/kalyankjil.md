# Kalyan Jewellers India Ltd. (KALYANKJIL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 413.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 55.20
- **Avg P&L per closed trade:** 13.80

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 12:15:00 | 559.45 | 700.49 | 701.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 13:15:00 | 539.85 | 698.89 | 700.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 14:15:00 | 578.25 | 560.26 | 612.61 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-10 09:15:00 | 525.35 | 558.75 | 606.17 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-02 14:15:00 | 509.70 | 471.25 | 506.39 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 09:15:00 | 554.10 | 514.71 | 514.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 556.85 | 517.26 | 515.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 545.45 | 545.50 | 534.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-10 15:15:00 | 547.80 | 545.51 | 535.09 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 536.35 | 545.23 | 535.21 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-12 09:15:00 | 534.15 | 544.94 | 535.21 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 14:15:00 | 509.70 | 557.57 | 557.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 504.55 | 548.13 | 552.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 11:15:00 | 528.50 | 520.39 | 533.29 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-18 12:15:00 | 512.95 | 520.48 | 532.83 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 502.70 | 490.55 | 505.52 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-23 13:15:00 | 499.10 | 490.84 | 505.44 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 504.75 | 491.70 | 504.97 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-27 13:15:00 | 505.55 | 491.83 | 504.97 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-10 09:15:00 | 525.35 | 2025-04-02 14:15:00 | 509.70 | EXIT_EMA400 | 15.65 |
| BUY | 2025-06-10 15:15:00 | 547.80 | 2025-06-12 09:15:00 | 534.15 | EXIT_EMA400 | -13.65 |
| SELL | 2025-09-18 12:15:00 | 512.95 | 2025-09-26 09:15:00 | 453.30 | TARGET | 59.65 |
| SELL | 2025-10-23 13:15:00 | 499.10 | 2025-10-27 13:15:00 | 505.55 | EXIT_EMA400 | -6.45 |
