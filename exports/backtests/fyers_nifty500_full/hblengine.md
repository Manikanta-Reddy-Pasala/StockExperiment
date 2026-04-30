# HBL Engineering Ltd. (HBLENGINE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 797.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 1 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 6 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -45.15
- **Avg P&L per closed trade:** -7.52

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 11:15:00 | 550.60 | 603.09 | 603.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 14:15:00 | 541.00 | 601.34 | 602.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 579.95 | 579.79 | 589.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 14:15:00 | 560.40 | 578.95 | 588.45 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-26 10:15:00 | 577.65 | 564.70 | 577.33 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 11:15:00 | 636.00 | 586.99 | 586.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 10:15:00 | 660.95 | 589.97 | 588.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-23 12:15:00 | 638.15 | 639.02 | 618.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-24 09:15:00 | 642.70 | 638.92 | 618.72 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-12-30 14:15:00 | 613.90 | 638.07 | 620.76 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 09:15:00 | 541.20 | 611.62 | 611.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 13:15:00 | 532.65 | 608.86 | 610.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 09:15:00 | 585.05 | 572.27 | 588.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-30 13:15:00 | 560.85 | 572.38 | 588.37 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-31 12:15:00 | 589.00 | 572.59 | 588.00 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 12:15:00 | 570.95 | 505.75 | 505.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 578.35 | 508.36 | 506.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 15:15:00 | 577.35 | 577.44 | 554.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 09:15:00 | 588.30 | 576.57 | 556.31 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-23 13:15:00 | 583.75 | 601.49 | 584.42 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 13:15:00 | 814.25 | 873.97 | 874.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 804.80 | 872.16 | 873.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 862.15 | 855.38 | 863.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-22 15:15:00 | 845.90 | 855.23 | 863.63 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-23 09:15:00 | 864.30 | 855.32 | 863.63 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 925.55 | 870.40 | 870.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 932.05 | 876.69 | 873.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 11:15:00 | 896.65 | 896.85 | 885.21 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 14:15:00 | 757.50 | 875.97 | 876.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 15:15:00 | 752.95 | 874.75 | 875.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 09:15:00 | 804.60 | 800.02 | 825.81 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-11 13:15:00 | 771.00 | 799.61 | 825.09 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 704.55 | 679.82 | 718.83 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-08 14:15:00 | 719.00 | 681.39 | 718.66 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 15:15:00 | 809.90 | 741.14 | 740.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 825.60 | 747.20 | 743.93 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-08 14:15:00 | 560.40 | 2024-11-26 10:15:00 | 577.65 | EXIT_EMA400 | -17.25 |
| BUY | 2024-12-24 09:15:00 | 642.70 | 2024-12-30 14:15:00 | 613.90 | EXIT_EMA400 | -28.80 |
| SELL | 2025-01-30 13:15:00 | 560.85 | 2025-01-31 12:15:00 | 589.00 | EXIT_EMA400 | -28.15 |
| BUY | 2025-06-24 09:15:00 | 588.30 | 2025-07-23 13:15:00 | 583.75 | EXIT_EMA400 | -4.55 |
| SELL | 2025-12-22 15:15:00 | 845.90 | 2025-12-23 09:15:00 | 864.30 | EXIT_EMA400 | -18.40 |
| SELL | 2026-02-11 13:15:00 | 771.00 | 2026-04-08 14:15:00 | 719.00 | EXIT_EMA400 | 52.00 |
