# Varun Beverages Ltd. (VBL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 513.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 3 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / EMA400 exits:** 3 / 5
- **Total realized P&L (per unit):** 33.19
- **Avg P&L per closed trade:** 4.15

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 15:15:00 | 603.36 | 615.32 | 615.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 09:15:00 | 596.56 | 615.13 | 615.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 10:15:00 | 617.88 | 614.02 | 614.69 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 13:15:00 | 635.74 | 615.39 | 615.35 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 10:15:00 | 607.52 | 615.67 | 615.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 14:15:00 | 599.98 | 614.66 | 615.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 612.58 | 611.65 | 613.48 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-10 14:15:00 | 607.78 | 611.69 | 613.45 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 607.78 | 611.69 | 613.45 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-09-11 09:15:00 | 624.00 | 611.77 | 613.47 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 14:15:00 | 645.20 | 615.08 | 615.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 655.30 | 615.78 | 615.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 09:15:00 | 625.25 | 628.70 | 622.96 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 548.00 | 618.71 | 618.75 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 10:15:00 | 642.65 | 607.46 | 607.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 13:15:00 | 644.45 | 613.12 | 610.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 12:15:00 | 625.70 | 625.95 | 618.43 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-18 14:15:00 | 632.25 | 626.05 | 618.55 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 628.90 | 626.13 | 618.67 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-12-19 10:15:00 | 631.75 | 626.18 | 618.73 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-12-20 14:15:00 | 613.75 | 626.09 | 619.09 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 15:15:00 | 569.80 | 618.03 | 618.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 09:15:00 | 560.30 | 617.46 | 617.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 14:15:00 | 585.05 | 572.75 | 590.28 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-06 10:15:00 | 569.10 | 574.45 | 589.74 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-18 10:15:00 | 529.20 | 499.85 | 528.43 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 505.40 | 486.38 | 486.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 516.40 | 491.15 | 488.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-28 12:15:00 | 501.05 | 501.40 | 495.60 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 09:15:00 | 473.60 | 492.21 | 492.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 10:15:00 | 470.30 | 491.99 | 492.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 11:15:00 | 458.95 | 456.67 | 468.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-28 09:15:00 | 453.70 | 458.41 | 467.40 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-29 09:15:00 | 467.90 | 458.25 | 467.00 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 475.20 | 467.84 | 467.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 12:15:00 | 475.80 | 467.92 | 467.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 13:15:00 | 469.75 | 470.22 | 469.12 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-18 14:15:00 | 473.75 | 470.20 | 469.15 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-19 10:15:00 | 468.65 | 470.25 | 469.19 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 439.90 | 476.66 | 476.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 13:15:00 | 437.75 | 475.53 | 476.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 15:15:00 | 463.90 | 462.77 | 467.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-25 10:15:00 | 456.40 | 462.65 | 467.88 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 13:15:00 | 433.00 | 418.32 | 433.92 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-10 14:15:00 | 430.95 | 418.44 | 433.90 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-15 09:15:00 | 442.30 | 419.61 | 433.82 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 501.70 | 444.50 | 444.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 13:15:00 | 506.85 | 445.12 | 444.60 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-10 14:15:00 | 607.78 | 2024-09-11 09:15:00 | 624.00 | EXIT_EMA400 | -16.22 |
| BUY | 2024-12-18 14:15:00 | 632.25 | 2024-12-20 14:15:00 | 613.75 | EXIT_EMA400 | -18.50 |
| BUY | 2024-12-19 10:15:00 | 631.75 | 2024-12-20 14:15:00 | 613.75 | EXIT_EMA400 | -18.00 |
| SELL | 2025-02-06 10:15:00 | 569.10 | 2025-02-14 09:15:00 | 507.17 | TARGET | 61.93 |
| SELL | 2025-10-28 09:15:00 | 453.70 | 2025-10-29 09:15:00 | 467.90 | EXIT_EMA400 | -14.20 |
| BUY | 2025-12-18 14:15:00 | 473.75 | 2025-12-19 10:15:00 | 468.65 | EXIT_EMA400 | -5.10 |
| SELL | 2026-02-25 10:15:00 | 456.40 | 2026-03-12 09:15:00 | 421.97 | TARGET | 34.43 |
| SELL | 2026-04-10 14:15:00 | 430.95 | 2026-04-13 09:15:00 | 422.09 | TARGET | 8.86 |
