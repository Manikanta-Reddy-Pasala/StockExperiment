# Saregama India Ltd (SAREGAMA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 342.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 6 |
| ENTRY1 | 4 |
| ENTRY2 | 4 |
| EXIT | 4 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 1 / 7
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -64.90
- **Avg P&L per closed trade:** -8.11

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 14:15:00 | 504.95 | 535.51 | 535.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-06 10:15:00 | 499.00 | 532.44 | 533.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 504.65 | 499.03 | 512.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-29 09:15:00 | 489.00 | 498.94 | 512.40 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 508.00 | 498.88 | 512.11 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-02 09:15:00 | 516.45 | 499.23 | 512.09 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 09:15:00 | 531.70 | 509.51 | 509.48 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 14:15:00 | 489.65 | 511.39 | 511.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 09:15:00 | 484.45 | 510.90 | 511.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 10:15:00 | 522.60 | 505.07 | 508.06 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 09:15:00 | 525.70 | 510.75 | 510.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-10 14:15:00 | 540.45 | 512.25 | 511.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 09:15:00 | 509.50 | 512.34 | 511.54 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 13:15:00 | 474.95 | 510.92 | 510.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 09:15:00 | 470.20 | 509.84 | 510.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 12:15:00 | 500.60 | 490.96 | 499.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-10 09:15:00 | 486.20 | 490.97 | 498.96 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 486.20 | 490.97 | 498.96 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-10 15:15:00 | 481.00 | 490.72 | 498.60 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 496.20 | 489.79 | 497.36 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-17 12:15:00 | 497.35 | 490.05 | 497.30 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 13:15:00 | 514.85 | 501.93 | 501.87 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 10:15:00 | 475.50 | 501.61 | 501.73 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 525.50 | 501.77 | 501.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 531.00 | 502.06 | 501.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 14:15:00 | 524.15 | 528.33 | 517.57 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-05 13:15:00 | 537.45 | 528.47 | 518.32 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-13 09:15:00 | 537.65 | 549.22 | 538.42 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 12:15:00 | 503.85 | 531.83 | 531.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 14:15:00 | 502.20 | 531.26 | 531.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 09:15:00 | 518.20 | 506.28 | 516.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-25 10:15:00 | 486.00 | 504.60 | 513.69 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 492.95 | 494.05 | 505.53 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-07 09:15:00 | 480.80 | 493.83 | 505.02 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 490.65 | 493.33 | 504.00 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-11 10:15:00 | 488.85 | 493.28 | 503.92 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 497.30 | 489.22 | 499.02 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-26 14:15:00 | 476.00 | 489.03 | 498.05 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-02 09:15:00 | 499.75 | 487.69 | 496.34 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-29 09:15:00 | 489.00 | 2024-12-02 09:15:00 | 516.45 | EXIT_EMA400 | -27.45 |
| SELL | 2025-03-10 09:15:00 | 486.20 | 2025-03-17 12:15:00 | 497.35 | EXIT_EMA400 | -11.15 |
| SELL | 2025-03-10 15:15:00 | 481.00 | 2025-03-17 12:15:00 | 497.35 | EXIT_EMA400 | -16.35 |
| BUY | 2025-05-05 13:15:00 | 537.45 | 2025-06-04 09:15:00 | 594.85 | TARGET | 57.40 |
| SELL | 2025-07-25 10:15:00 | 486.00 | 2025-09-02 09:15:00 | 499.75 | EXIT_EMA400 | -13.75 |
| SELL | 2025-08-07 09:15:00 | 480.80 | 2025-09-02 09:15:00 | 499.75 | EXIT_EMA400 | -18.95 |
| SELL | 2025-08-11 10:15:00 | 488.85 | 2025-09-02 09:15:00 | 499.75 | EXIT_EMA400 | -10.90 |
| SELL | 2025-08-26 14:15:00 | 476.00 | 2025-09-02 09:15:00 | 499.75 | EXIT_EMA400 | -23.75 |
