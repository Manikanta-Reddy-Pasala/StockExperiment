# Berger Paints India Ltd. (BERGEPAINT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 474.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / EMA400 exits:** 3 / 3
- **Total realized P&L (per unit):** 29.31
- **Avg P&L per closed trade:** 4.89

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 15:15:00 | 543.00 | 568.01 | 568.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 11:15:00 | 542.40 | 565.86 | 566.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 468.80 | 464.91 | 487.14 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-10 13:15:00 | 461.35 | 465.06 | 486.03 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 479.65 | 466.04 | 481.11 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-23 10:15:00 | 482.75 | 466.21 | 481.11 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 15:15:00 | 501.50 | 483.41 | 483.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 09:15:00 | 506.75 | 485.01 | 484.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 10:15:00 | 485.30 | 488.49 | 486.16 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-03-17 09:15:00 | 497.40 | 488.18 | 486.14 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 497.40 | 488.18 | 486.14 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-03-17 10:15:00 | 500.15 | 488.30 | 486.21 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 492.20 | 495.97 | 491.40 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-04-02 09:15:00 | 494.40 | 495.91 | 491.44 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-19 12:15:00 | 547.00 | 559.76 | 547.29 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 536.50 | 560.20 | 560.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 530.65 | 556.78 | 558.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 13:15:00 | 544.70 | 543.98 | 549.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-16 14:15:00 | 536.80 | 544.24 | 549.26 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-03 14:15:00 | 543.00 | 534.85 | 542.09 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 15:15:00 | 581.00 | 541.82 | 541.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 10:15:00 | 587.00 | 547.28 | 544.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 10:15:00 | 560.30 | 560.92 | 553.38 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 10:15:00 | 537.80 | 549.63 | 549.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 535.25 | 548.00 | 548.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 14:15:00 | 431.00 | 430.76 | 454.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-07 09:15:00 | 426.95 | 430.76 | 454.00 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 452.15 | 431.78 | 452.21 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-10 11:15:00 | 454.50 | 432.20 | 452.22 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-10 13:15:00 | 461.35 | 2025-01-23 10:15:00 | 482.75 | EXIT_EMA400 | -21.40 |
| BUY | 2025-04-02 09:15:00 | 494.40 | 2025-04-02 11:15:00 | 503.27 | TARGET | 8.87 |
| BUY | 2025-03-17 09:15:00 | 497.40 | 2025-04-08 11:15:00 | 531.18 | TARGET | 33.78 |
| BUY | 2025-03-17 10:15:00 | 500.15 | 2025-04-09 09:15:00 | 541.97 | TARGET | 41.82 |
| SELL | 2025-09-16 14:15:00 | 536.80 | 2025-10-03 14:15:00 | 543.00 | EXIT_EMA400 | -6.20 |
| SELL | 2026-04-07 09:15:00 | 426.95 | 2026-04-10 11:15:00 | 454.50 | EXIT_EMA400 | -27.55 |
