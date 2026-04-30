# Jindal Stainless Ltd. (JSL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 769.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / EMA400 exits:** 3 / 5
- **Total realized P&L (per unit):** 64.24
- **Avg P&L per closed trade:** 8.03

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 12:15:00 | 677.80 | 752.96 | 753.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 09:15:00 | 669.35 | 749.97 | 751.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 09:15:00 | 729.05 | 726.47 | 737.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-04 09:15:00 | 717.15 | 733.14 | 738.02 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 728.10 | 731.54 | 736.86 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-09 10:15:00 | 721.10 | 731.26 | 736.51 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-09-10 10:15:00 | 737.90 | 730.77 | 736.08 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 766.20 | 740.12 | 740.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 10:15:00 | 779.00 | 740.51 | 740.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 14:15:00 | 756.80 | 757.64 | 750.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-08 12:15:00 | 763.70 | 757.59 | 750.57 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-08 15:15:00 | 749.30 | 757.50 | 750.63 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 15:15:00 | 652.00 | 746.71 | 747.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 645.65 | 734.65 | 740.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 13:15:00 | 712.85 | 712.06 | 726.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-07 10:15:00 | 708.35 | 712.13 | 725.95 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 724.10 | 712.52 | 725.88 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-07 15:15:00 | 718.00 | 712.58 | 725.84 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-12-03 09:15:00 | 710.80 | 696.09 | 710.30 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 10:15:00 | 752.00 | 719.84 | 719.71 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 09:15:00 | 690.55 | 720.54 | 720.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 10:15:00 | 685.30 | 720.19 | 720.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-21 12:15:00 | 630.70 | 623.90 | 648.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-24 09:15:00 | 615.75 | 623.86 | 647.77 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-06 09:15:00 | 642.85 | 614.24 | 637.15 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 10:15:00 | 646.45 | 606.49 | 606.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 660.70 | 613.28 | 609.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 11:15:00 | 663.50 | 667.10 | 645.39 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-23 09:15:00 | 670.35 | 666.71 | 646.45 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 669.70 | 682.05 | 668.75 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-24 12:15:00 | 666.00 | 681.89 | 668.73 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 14:15:00 | 759.05 | 784.17 | 784.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 09:15:00 | 751.85 | 783.59 | 783.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 11:15:00 | 782.20 | 779.59 | 781.81 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 15:15:00 | 809.90 | 783.80 | 783.78 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 777.40 | 783.74 | 783.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 13:15:00 | 773.80 | 783.62 | 783.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 15:15:00 | 785.55 | 783.62 | 783.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-04 09:15:00 | 759.80 | 783.38 | 783.57 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-08 09:15:00 | 751.00 | 734.22 | 750.64 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 11:15:00 | 771.10 | 760.62 | 760.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 783.25 | 761.68 | 761.12 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-04 09:15:00 | 717.15 | 2024-09-10 10:15:00 | 737.90 | EXIT_EMA400 | -20.75 |
| SELL | 2024-09-09 10:15:00 | 721.10 | 2024-09-10 10:15:00 | 737.90 | EXIT_EMA400 | -16.80 |
| BUY | 2024-10-08 12:15:00 | 763.70 | 2024-10-08 15:15:00 | 749.30 | EXIT_EMA400 | -14.40 |
| SELL | 2024-11-07 15:15:00 | 718.00 | 2024-11-12 15:15:00 | 694.48 | TARGET | 23.52 |
| SELL | 2024-11-07 10:15:00 | 708.35 | 2024-11-21 10:15:00 | 655.54 | TARGET | 52.81 |
| SELL | 2025-02-24 09:15:00 | 615.75 | 2025-03-06 09:15:00 | 642.85 | EXIT_EMA400 | -27.10 |
| BUY | 2025-06-23 09:15:00 | 670.35 | 2025-07-24 12:15:00 | 666.00 | EXIT_EMA400 | -4.35 |
| SELL | 2026-03-04 09:15:00 | 759.80 | 2026-03-16 10:15:00 | 688.49 | TARGET | 71.31 |
