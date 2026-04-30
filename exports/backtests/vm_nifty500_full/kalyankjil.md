# Kalyan Jewellers India Ltd. (KALYANKJIL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 412.75
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 2 / 1
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 61.91
- **Avg P&L per closed trade:** 20.64

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 12:15:00 | 559.45 | 700.51 | 701.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 13:15:00 | 539.90 | 698.91 | 700.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 14:15:00 | 578.25 | 563.66 | 616.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-10 09:15:00 | 525.35 | 561.44 | 609.30 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-02 14:15:00 | 509.65 | 471.48 | 507.32 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 11:15:00 | 549.55 | 515.48 | 515.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 557.15 | 517.31 | 516.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 545.45 | 545.54 | 535.04 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-10 15:15:00 | 547.80 | 545.54 | 535.30 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 536.35 | 545.29 | 535.42 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-12 09:15:00 | 534.25 | 544.99 | 535.42 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 14:15:00 | 509.75 | 557.59 | 557.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 504.85 | 548.15 | 552.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 11:15:00 | 528.60 | 520.40 | 533.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-18 12:15:00 | 512.95 | 520.50 | 532.87 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 504.75 | 491.71 | 504.99 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-27 13:15:00 | 505.55 | 491.85 | 504.99 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-10 09:15:00 | 525.35 | 2025-04-02 14:15:00 | 509.65 | EXIT_EMA400 | 15.70 |
| BUY | 2025-06-10 15:15:00 | 547.80 | 2025-06-12 09:15:00 | 534.25 | EXIT_EMA400 | -13.55 |
| SELL | 2025-09-18 12:15:00 | 512.95 | 2025-09-26 09:15:00 | 453.19 | TARGET | 59.76 |
