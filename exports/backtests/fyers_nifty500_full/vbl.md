# Varun Beverages Ltd. (VBL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 516.30
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
| ENTRY1 | 7 |
| ENTRY2 | 3 |
| EXIT | 7 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 4 / 6
- **Target hits / EMA400 exits:** 4 / 6
- **Total realized P&L (per unit):** 82.59
- **Avg P&L per closed trade:** 8.26

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 13:15:00 | 571.18 | 617.51 | 617.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-16 14:15:00 | 568.40 | 617.02 | 617.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 10:15:00 | 617.88 | 613.98 | 615.80 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 13:15:00 | 632.78 | 617.43 | 617.41 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 11:15:00 | 598.44 | 617.43 | 617.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 12:15:00 | 596.82 | 612.72 | 614.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 612.58 | 611.67 | 614.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-10 14:15:00 | 607.96 | 611.70 | 614.13 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-09-11 09:15:00 | 624.00 | 611.78 | 614.15 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-09-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 11:15:00 | 652.95 | 616.59 | 616.45 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 09:15:00 | 552.25 | 619.45 | 619.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 547.90 | 618.74 | 619.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 14:15:00 | 611.80 | 607.19 | 612.48 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-17 09:15:00 | 597.25 | 607.10 | 612.20 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-10-23 09:15:00 | 618.40 | 602.26 | 608.95 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 11:15:00 | 640.65 | 607.84 | 607.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 12:15:00 | 643.60 | 612.83 | 610.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 12:15:00 | 625.70 | 625.95 | 618.49 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-18 14:15:00 | 632.45 | 626.05 | 618.61 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 628.90 | 626.13 | 618.73 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-12-19 10:15:00 | 631.75 | 626.19 | 618.79 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-12-20 14:15:00 | 613.75 | 626.10 | 619.15 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 15:15:00 | 569.80 | 618.03 | 618.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 09:15:00 | 560.30 | 617.46 | 617.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 13:15:00 | 572.65 | 571.95 | 589.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-01 14:15:00 | 559.00 | 571.82 | 589.83 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 584.95 | 572.07 | 589.17 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-06 09:15:00 | 574.50 | 573.73 | 588.87 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-18 10:15:00 | 529.30 | 499.74 | 528.05 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 505.50 | 486.35 | 486.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 516.60 | 491.12 | 488.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-28 12:15:00 | 501.05 | 501.41 | 495.59 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 09:15:00 | 473.60 | 492.22 | 492.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 10:15:00 | 470.40 | 492.00 | 492.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 11:15:00 | 458.75 | 456.68 | 468.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-28 09:15:00 | 453.75 | 458.42 | 467.40 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-29 09:15:00 | 467.90 | 458.27 | 467.01 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 475.80 | 467.90 | 467.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 14:15:00 | 478.55 | 468.05 | 467.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 13:15:00 | 469.75 | 470.20 | 469.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-18 14:15:00 | 473.80 | 470.17 | 469.13 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-19 10:15:00 | 468.70 | 470.21 | 469.17 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 15:15:00 | 444.10 | 476.68 | 476.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 441.95 | 476.33 | 476.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 15:15:00 | 461.80 | 461.78 | 467.05 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-27 09:15:00 | 456.65 | 461.73 | 467.00 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 13:15:00 | 433.00 | 418.27 | 433.83 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-10 14:15:00 | 430.80 | 418.40 | 433.82 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-15 09:15:00 | 442.30 | 419.57 | 433.74 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 495.90 | 444.13 | 444.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 501.75 | 445.68 | 444.79 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-10 14:15:00 | 607.96 | 2024-09-11 09:15:00 | 624.00 | EXIT_EMA400 | -16.04 |
| SELL | 2024-10-17 09:15:00 | 597.25 | 2024-10-23 09:15:00 | 618.40 | EXIT_EMA400 | -21.15 |
| BUY | 2024-12-18 14:15:00 | 632.45 | 2024-12-20 14:15:00 | 613.75 | EXIT_EMA400 | -18.70 |
| BUY | 2024-12-19 10:15:00 | 631.75 | 2024-12-20 14:15:00 | 613.75 | EXIT_EMA400 | -18.00 |
| SELL | 2025-02-06 09:15:00 | 574.50 | 2025-02-11 09:15:00 | 531.38 | TARGET | 43.12 |
| SELL | 2025-02-01 14:15:00 | 559.00 | 2025-02-18 14:15:00 | 466.50 | TARGET | 92.50 |
| SELL | 2025-10-28 09:15:00 | 453.75 | 2025-10-29 09:15:00 | 467.90 | EXIT_EMA400 | -14.15 |
| BUY | 2025-12-18 14:15:00 | 473.80 | 2025-12-19 10:15:00 | 468.70 | EXIT_EMA400 | -5.10 |
| SELL | 2026-02-27 09:15:00 | 456.65 | 2026-03-12 09:15:00 | 425.59 | TARGET | 31.06 |
| SELL | 2026-04-10 14:15:00 | 430.80 | 2026-04-13 09:15:00 | 421.74 | TARGET | 9.06 |
