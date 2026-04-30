# Minda Corporation Ltd. (MINDACORP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 520.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 4 |
| ENTRY1 | 8 |
| ENTRY2 | 3 |
| EXIT | 8 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 2 / 9
- **Target hits / EMA400 exits:** 2 / 9
- **Total realized P&L (per unit):** -120.51
- **Avg P&L per closed trade:** -10.96

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 15:15:00 | 506.00 | 538.39 | 538.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 499.85 | 535.84 | 537.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 11:15:00 | 532.95 | 532.46 | 535.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-07 14:15:00 | 521.60 | 531.98 | 534.84 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-06 13:15:00 | 521.45 | 505.90 | 515.91 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 565.35 | 517.90 | 517.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 13:15:00 | 571.50 | 518.91 | 518.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 10:15:00 | 539.95 | 542.35 | 531.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-31 11:15:00 | 566.60 | 542.04 | 533.14 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-02-11 09:15:00 | 537.25 | 552.66 | 541.30 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 485.00 | 534.63 | 534.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 12:15:00 | 479.40 | 533.20 | 533.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 10:15:00 | 528.00 | 524.71 | 529.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-07 14:15:00 | 519.00 | 524.61 | 528.98 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-10 09:15:00 | 531.50 | 524.61 | 528.94 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 549.80 | 514.52 | 514.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 560.40 | 515.69 | 514.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 11:15:00 | 529.00 | 530.47 | 523.55 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-18 09:15:00 | 543.15 | 530.51 | 523.74 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-19 10:15:00 | 519.25 | 530.91 | 524.22 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 15:15:00 | 516.00 | 520.26 | 520.26 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 09:15:00 | 533.50 | 520.31 | 520.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 10:15:00 | 539.00 | 520.50 | 520.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 12:15:00 | 520.95 | 521.90 | 521.12 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 12:15:00 | 508.80 | 520.37 | 520.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 10:15:00 | 504.60 | 519.30 | 519.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 508.55 | 496.99 | 506.35 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-21 11:15:00 | 504.50 | 499.94 | 506.91 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 504.50 | 499.94 | 506.91 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-21 14:15:00 | 499.50 | 500.00 | 506.83 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-08-22 11:15:00 | 508.65 | 500.14 | 506.76 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 11:15:00 | 529.35 | 509.16 | 509.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 09:15:00 | 540.30 | 510.33 | 509.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 563.15 | 563.38 | 548.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-03 09:15:00 | 578.45 | 563.53 | 548.87 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-09 09:15:00 | 565.30 | 585.97 | 573.87 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 516.55 | 577.63 | 577.67 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 11:15:00 | 593.20 | 573.74 | 573.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 09:15:00 | 599.30 | 574.70 | 574.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 573.00 | 578.32 | 576.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-20 14:15:00 | 584.35 | 578.27 | 576.33 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 578.15 | 578.27 | 576.36 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-02-23 11:15:00 | 574.45 | 578.24 | 576.35 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 536.05 | 574.76 | 574.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 531.40 | 573.33 | 574.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 15:15:00 | 525.85 | 524.95 | 543.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-27 10:15:00 | 515.80 | 525.06 | 542.99 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 521.10 | 524.77 | 542.49 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-03-30 09:15:00 | 504.25 | 524.62 | 542.24 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 14:15:00 | 533.00 | 519.43 | 536.13 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-09 09:15:00 | 522.20 | 519.56 | 536.03 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-17 11:15:00 | 537.25 | 518.19 | 532.50 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-07 14:15:00 | 521.60 | 2024-11-13 14:15:00 | 481.88 | TARGET | 39.72 |
| BUY | 2025-01-31 11:15:00 | 566.60 | 2025-02-11 09:15:00 | 537.25 | EXIT_EMA400 | -29.35 |
| SELL | 2025-03-07 14:15:00 | 519.00 | 2025-03-10 09:15:00 | 531.50 | EXIT_EMA400 | -12.50 |
| BUY | 2025-06-18 09:15:00 | 543.15 | 2025-06-19 10:15:00 | 519.25 | EXIT_EMA400 | -23.90 |
| SELL | 2025-08-21 11:15:00 | 504.50 | 2025-08-22 09:15:00 | 497.28 | TARGET | 7.22 |
| SELL | 2025-08-21 14:15:00 | 499.50 | 2025-08-22 11:15:00 | 508.65 | EXIT_EMA400 | -9.15 |
| BUY | 2025-11-03 09:15:00 | 578.45 | 2025-12-09 09:15:00 | 565.30 | EXIT_EMA400 | -13.15 |
| BUY | 2026-02-20 14:15:00 | 584.35 | 2026-02-23 11:15:00 | 574.45 | EXIT_EMA400 | -9.90 |
| SELL | 2026-03-27 10:15:00 | 515.80 | 2026-04-17 11:15:00 | 537.25 | EXIT_EMA400 | -21.45 |
| SELL | 2026-03-30 09:15:00 | 504.25 | 2026-04-17 11:15:00 | 537.25 | EXIT_EMA400 | -33.00 |
| SELL | 2026-04-09 09:15:00 | 522.20 | 2026-04-17 11:15:00 | 537.25 | EXIT_EMA400 | -15.05 |
