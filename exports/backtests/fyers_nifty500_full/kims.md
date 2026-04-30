# Krishna Institute of Medical Sciences Ltd. (KIMS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 663.45
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -20.25
- **Avg P&L per closed trade:** -3.37

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 543.35 | 596.03 | 596.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 12:15:00 | 540.15 | 594.95 | 595.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 15:15:00 | 552.00 | 551.20 | 568.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-13 10:15:00 | 542.95 | 551.06 | 568.46 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-13 14:15:00 | 570.00 | 550.88 | 568.03 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 620.00 | 579.68 | 579.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 12:15:00 | 625.25 | 580.52 | 579.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 13:15:00 | 586.65 | 588.68 | 584.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-04 14:15:00 | 593.10 | 588.72 | 584.53 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 593.10 | 588.72 | 584.53 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-04-07 09:15:00 | 559.45 | 588.47 | 584.44 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 15:15:00 | 701.45 | 722.68 | 722.75 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 727.35 | 722.71 | 722.69 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 09:15:00 | 718.25 | 722.67 | 722.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 716.60 | 722.61 | 722.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 12:15:00 | 725.00 | 722.62 | 722.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-27 10:15:00 | 719.40 | 722.58 | 722.63 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 718.85 | 722.54 | 722.61 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-27 12:15:00 | 716.25 | 722.48 | 722.58 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-28 09:15:00 | 728.65 | 722.40 | 722.54 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 14:15:00 | 732.85 | 722.73 | 722.69 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 708.80 | 722.64 | 722.68 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 727.65 | 722.75 | 722.74 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 10:15:00 | 704.50 | 722.59 | 722.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 11:15:00 | 698.80 | 722.35 | 722.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 700.40 | 691.75 | 703.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-08 14:15:00 | 678.30 | 693.84 | 702.25 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-09 11:15:00 | 646.50 | 616.08 | 636.43 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 699.95 | 650.73 | 650.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 710.70 | 653.05 | 651.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 683.75 | 684.66 | 671.64 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 634.40 | 663.34 | 663.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 619.90 | 662.36 | 662.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 671.80 | 651.98 | 657.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 09:15:00 | 650.65 | 654.13 | 657.79 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-13 10:15:00 | 660.40 | 654.19 | 657.80 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 09:15:00 | 680.85 | 660.77 | 660.68 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 646.55 | 660.55 | 660.60 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 674.65 | 660.73 | 660.69 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-13 10:15:00 | 542.95 | 2025-03-13 14:15:00 | 570.00 | EXIT_EMA400 | -27.05 |
| BUY | 2025-04-04 14:15:00 | 593.10 | 2025-04-07 09:15:00 | 559.45 | EXIT_EMA400 | -33.65 |
| SELL | 2025-10-27 10:15:00 | 719.40 | 2025-10-28 09:15:00 | 728.65 | EXIT_EMA400 | -9.25 |
| SELL | 2025-10-27 12:15:00 | 716.25 | 2025-10-28 09:15:00 | 728.65 | EXIT_EMA400 | -12.40 |
| SELL | 2025-12-08 14:15:00 | 678.30 | 2025-12-30 09:15:00 | 606.45 | TARGET | 71.85 |
| SELL | 2026-04-13 09:15:00 | 650.65 | 2026-04-13 10:15:00 | 660.40 | EXIT_EMA400 | -9.75 |
