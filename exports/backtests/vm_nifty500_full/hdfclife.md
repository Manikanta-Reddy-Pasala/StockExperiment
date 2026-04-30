# HDFC Life Insurance Company Ltd. (HDFCLIFE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 586.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 5 |
| ENTRY1 | 10 |
| ENTRY2 | 1 |
| EXIT | 10 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 1 / 10
- **Target hits / EMA400 exits:** 1 / 10
- **Total realized P&L (per unit):** -102.45
- **Avg P&L per closed trade:** -9.31

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-10 15:15:00 | 621.25 | 638.11 | 638.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 09:15:00 | 613.05 | 635.85 | 636.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 09:15:00 | 626.65 | 625.74 | 630.14 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-13 12:15:00 | 619.05 | 625.57 | 629.84 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-11-15 09:15:00 | 630.65 | 625.49 | 629.72 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-22 10:15:00 | 670.80 | 633.29 | 633.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-23 09:15:00 | 672.80 | 635.40 | 634.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 14:15:00 | 665.55 | 665.74 | 654.44 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-20 10:15:00 | 671.00 | 665.84 | 654.66 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2023-12-20 13:15:00 | 652.00 | 665.68 | 654.75 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 14:15:00 | 614.20 | 649.06 | 649.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 11:15:00 | 610.95 | 647.66 | 648.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-07 09:15:00 | 612.00 | 610.07 | 625.45 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-08 13:15:00 | 595.25 | 609.50 | 624.33 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-03-04 12:15:00 | 615.35 | 591.91 | 606.35 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 11:15:00 | 632.40 | 613.68 | 613.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 12:15:00 | 634.00 | 613.88 | 613.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 09:15:00 | 619.95 | 620.49 | 617.61 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 09:15:00 | 598.15 | 615.36 | 615.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 13:15:00 | 595.40 | 614.67 | 615.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 10:15:00 | 576.10 | 575.45 | 588.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-29 09:15:00 | 567.40 | 575.42 | 588.12 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-06-13 09:15:00 | 588.55 | 567.10 | 578.82 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-07-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 14:15:00 | 607.45 | 585.31 | 585.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 09:15:00 | 612.40 | 585.79 | 585.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 10:15:00 | 711.75 | 715.89 | 683.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-20 09:15:00 | 718.75 | 710.22 | 688.19 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-25 09:15:00 | 702.50 | 722.22 | 708.44 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 14:15:00 | 686.05 | 704.75 | 704.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 14:15:00 | 680.70 | 702.18 | 703.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-16 09:15:00 | 653.10 | 625.04 | 646.15 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-21 10:15:00 | 626.55 | 627.70 | 645.37 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 636.55 | 624.52 | 639.27 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-31 15:15:00 | 639.55 | 625.30 | 639.23 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 10:15:00 | 681.50 | 634.11 | 633.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 09:15:00 | 692.35 | 646.97 | 640.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 12:15:00 | 656.90 | 658.18 | 647.84 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 13:15:00 | 663.15 | 658.23 | 647.92 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 760.15 | 776.57 | 756.61 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-14 10:15:00 | 760.65 | 776.41 | 756.63 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 757.25 | 775.41 | 756.90 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-15 12:15:00 | 752.85 | 775.19 | 756.88 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 10:15:00 | 743.15 | 766.39 | 766.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 740.00 | 763.13 | 764.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 15:15:00 | 763.35 | 761.90 | 763.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-16 09:15:00 | 741.50 | 761.70 | 763.86 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 756.55 | 754.15 | 759.14 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-29 12:15:00 | 760.05 | 754.26 | 759.12 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 785.45 | 759.54 | 759.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 786.70 | 759.81 | 759.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 11:15:00 | 760.75 | 761.94 | 760.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-01 14:15:00 | 767.55 | 761.98 | 760.85 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 767.55 | 761.98 | 760.85 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-02 09:15:00 | 757.30 | 761.98 | 760.86 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 747.05 | 761.00 | 761.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 743.40 | 760.68 | 760.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 09:15:00 | 761.70 | 758.12 | 759.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-09 12:15:00 | 750.80 | 759.99 | 760.29 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-12 09:15:00 | 761.25 | 759.70 | 760.14 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-13 12:15:00 | 619.05 | 2023-11-15 09:15:00 | 630.65 | EXIT_EMA400 | -11.60 |
| BUY | 2023-12-20 10:15:00 | 671.00 | 2023-12-20 13:15:00 | 652.00 | EXIT_EMA400 | -19.00 |
| SELL | 2024-02-08 13:15:00 | 595.25 | 2024-03-04 12:15:00 | 615.35 | EXIT_EMA400 | -20.10 |
| SELL | 2024-05-29 09:15:00 | 567.40 | 2024-06-13 09:15:00 | 588.55 | EXIT_EMA400 | -21.15 |
| BUY | 2024-09-20 09:15:00 | 718.75 | 2024-10-25 09:15:00 | 702.50 | EXIT_EMA400 | -16.25 |
| SELL | 2025-01-21 10:15:00 | 626.55 | 2025-01-31 15:15:00 | 639.55 | EXIT_EMA400 | -13.00 |
| BUY | 2025-04-07 13:15:00 | 663.15 | 2025-04-16 09:15:00 | 708.85 | TARGET | 45.70 |
| BUY | 2025-07-14 10:15:00 | 760.65 | 2025-07-15 12:15:00 | 752.85 | EXIT_EMA400 | -7.80 |
| SELL | 2025-10-16 09:15:00 | 741.50 | 2025-10-29 12:15:00 | 760.05 | EXIT_EMA400 | -18.55 |
| BUY | 2025-12-01 14:15:00 | 767.55 | 2025-12-02 09:15:00 | 757.30 | EXIT_EMA400 | -10.25 |
| SELL | 2026-01-09 12:15:00 | 750.80 | 2026-01-12 09:15:00 | 761.25 | EXIT_EMA400 | -10.45 |
