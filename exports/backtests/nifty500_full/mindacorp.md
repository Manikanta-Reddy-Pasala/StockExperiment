# Minda Corporation Ltd. (MINDACORP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 520.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 5 |
| ENTRY1 | 9 |
| ENTRY2 | 3 |
| EXIT | 9 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 3 / 9
- **Target hits / EMA400 exits:** 3 / 9
- **Total realized P&L (per unit):** -71.87
- **Avg P&L per closed trade:** -5.99

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 15:15:00 | 506.00 | 538.39 | 538.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 499.85 | 536.18 | 537.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 11:15:00 | 532.95 | 532.76 | 535.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-07 14:15:00 | 521.60 | 532.25 | 535.07 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-06 13:15:00 | 518.10 | 505.97 | 516.03 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 12:15:00 | 565.95 | 518.33 | 518.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 13:15:00 | 571.50 | 518.86 | 518.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 10:15:00 | 539.95 | 542.30 | 531.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-31 11:15:00 | 566.60 | 542.04 | 533.16 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-02-11 09:15:00 | 537.25 | 551.73 | 540.47 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 10:15:00 | 492.00 | 533.81 | 533.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 12:15:00 | 479.40 | 532.81 | 533.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 10:15:00 | 528.00 | 524.44 | 528.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-07 14:15:00 | 518.40 | 524.34 | 528.53 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-10 09:15:00 | 531.50 | 524.37 | 528.50 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 11:15:00 | 537.80 | 529.28 | 529.24 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 516.75 | 529.17 | 529.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 484.10 | 528.19 | 528.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-21 11:15:00 | 516.45 | 516.20 | 521.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-25 09:15:00 | 506.15 | 516.74 | 521.25 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 504.50 | 500.47 | 509.85 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-05-19 09:15:00 | 514.20 | 501.05 | 509.56 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 09:15:00 | 549.60 | 514.21 | 514.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 560.40 | 515.73 | 514.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 11:15:00 | 529.00 | 530.55 | 523.56 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-18 09:15:00 | 543.15 | 530.59 | 523.75 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-19 10:15:00 | 519.25 | 530.98 | 524.23 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 10:15:00 | 517.85 | 520.23 | 520.24 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 09:15:00 | 533.15 | 520.32 | 520.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 10:15:00 | 539.00 | 520.51 | 520.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 12:15:00 | 520.90 | 521.91 | 521.13 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 12:15:00 | 508.80 | 520.39 | 520.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 10:15:00 | 504.60 | 519.33 | 519.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 508.55 | 497.02 | 506.38 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-21 11:15:00 | 504.50 | 499.99 | 506.94 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 504.50 | 499.99 | 506.94 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-21 13:15:00 | 502.65 | 500.05 | 506.90 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-08-22 11:15:00 | 508.65 | 500.18 | 506.80 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 11:15:00 | 529.35 | 509.17 | 509.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 09:15:00 | 540.30 | 510.33 | 509.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 563.20 | 563.38 | 548.65 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-03 09:15:00 | 578.45 | 563.53 | 548.87 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-09 09:15:00 | 565.30 | 586.00 | 573.89 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 516.55 | 577.65 | 577.70 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 13:15:00 | 594.10 | 574.25 | 574.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 09:15:00 | 599.30 | 574.84 | 574.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 573.60 | 578.43 | 576.49 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-20 14:15:00 | 584.35 | 578.38 | 576.53 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 578.15 | 578.38 | 576.55 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-02-23 11:15:00 | 574.45 | 578.34 | 576.54 | Close below EMA400 |

### Cycle 13 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 533.80 | 575.14 | 575.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 531.40 | 573.32 | 574.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 526.00 | 524.93 | 543.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-27 10:15:00 | 515.80 | 525.05 | 543.05 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 521.10 | 524.76 | 542.55 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-03-30 09:15:00 | 504.25 | 524.61 | 542.29 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 14:15:00 | 533.00 | 519.49 | 536.22 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-09 09:15:00 | 522.20 | 519.63 | 536.12 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-17 11:15:00 | 537.25 | 518.21 | 532.56 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-07 14:15:00 | 521.60 | 2024-11-13 14:15:00 | 481.20 | TARGET | 40.40 |
| BUY | 2025-01-31 11:15:00 | 566.60 | 2025-02-11 09:15:00 | 537.25 | EXIT_EMA400 | -29.35 |
| SELL | 2025-03-07 14:15:00 | 518.40 | 2025-03-10 09:15:00 | 531.50 | EXIT_EMA400 | -13.10 |
| SELL | 2025-04-25 09:15:00 | 506.15 | 2025-05-09 09:15:00 | 460.84 | TARGET | 45.31 |
| BUY | 2025-06-18 09:15:00 | 543.15 | 2025-06-19 10:15:00 | 519.25 | EXIT_EMA400 | -23.90 |
| SELL | 2025-08-21 11:15:00 | 504.50 | 2025-08-22 09:15:00 | 497.19 | TARGET | 7.31 |
| SELL | 2025-08-21 13:15:00 | 502.65 | 2025-08-22 11:15:00 | 508.65 | EXIT_EMA400 | -6.00 |
| BUY | 2025-11-03 09:15:00 | 578.45 | 2025-12-09 09:15:00 | 565.30 | EXIT_EMA400 | -13.15 |
| BUY | 2026-02-20 14:15:00 | 584.35 | 2026-02-23 11:15:00 | 574.45 | EXIT_EMA400 | -9.90 |
| SELL | 2026-03-27 10:15:00 | 515.80 | 2026-04-17 11:15:00 | 537.25 | EXIT_EMA400 | -21.45 |
| SELL | 2026-03-30 09:15:00 | 504.25 | 2026-04-17 11:15:00 | 537.25 | EXIT_EMA400 | -33.00 |
| SELL | 2026-04-09 09:15:00 | 522.20 | 2026-04-17 11:15:00 | 537.25 | EXIT_EMA400 | -15.05 |
