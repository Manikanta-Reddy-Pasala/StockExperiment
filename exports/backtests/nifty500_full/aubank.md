# AU Small Finance Bank Ltd. (AUBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 1015.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT3 | 6 |
| ENTRY1 | 6 |
| ENTRY2 | 3 |
| EXIT | 6 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 0 / 9
- **Target hits / EMA400 exits:** 0 / 9
- **Total realized P&L (per unit):** -210.60
- **Avg P&L per closed trade:** -23.40

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 13:15:00 | 743.50 | 716.48 | 716.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 09:15:00 | 750.85 | 717.34 | 716.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 14:15:00 | 730.95 | 733.75 | 726.98 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-20 09:15:00 | 758.35 | 733.99 | 727.16 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 10:15:00 | 755.75 | 767.53 | 752.48 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-01-17 11:15:00 | 747.40 | 767.33 | 752.45 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 12:15:00 | 626.90 | 741.76 | 741.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-06 09:15:00 | 618.80 | 711.61 | 725.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 14:15:00 | 589.40 | 585.27 | 618.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-03 10:15:00 | 579.05 | 585.76 | 617.53 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-04 09:15:00 | 634.50 | 586.43 | 616.92 | Close above EMA400 |

### Cycle 3 — BUY (started 2024-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 15:15:00 | 648.30 | 624.08 | 623.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 13:15:00 | 653.20 | 626.47 | 625.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 618.05 | 628.52 | 626.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-05 10:15:00 | 648.15 | 628.84 | 626.58 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 653.00 | 649.74 | 639.90 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-06-24 10:15:00 | 671.70 | 649.96 | 640.06 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-07-08 11:15:00 | 646.35 | 661.91 | 650.22 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 13:15:00 | 628.30 | 645.29 | 645.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 14:15:00 | 625.35 | 645.10 | 645.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-22 09:15:00 | 635.00 | 632.29 | 637.81 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-23 11:15:00 | 629.20 | 632.24 | 637.54 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 632.55 | 632.06 | 637.32 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-08-28 15:15:00 | 628.30 | 632.41 | 637.01 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 635.00 | 632.44 | 636.97 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-08-29 11:15:00 | 648.95 | 632.60 | 637.03 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 13:15:00 | 683.30 | 641.23 | 641.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 12:15:00 | 691.90 | 643.50 | 642.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 13:15:00 | 707.65 | 712.97 | 690.78 | EMA200 retest candle locked |

### Cycle 6 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 619.75 | 680.35 | 680.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 12:15:00 | 616.20 | 679.12 | 680.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 14:15:00 | 587.50 | 577.64 | 602.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-14 09:15:00 | 545.05 | 581.81 | 588.60 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 10:15:00 | 552.75 | 537.88 | 554.76 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-24 11:15:00 | 548.30 | 537.99 | 554.73 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 553.00 | 538.64 | 554.64 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-25 10:15:00 | 556.70 | 538.82 | 554.66 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 15:15:00 | 613.80 | 559.99 | 559.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 09:15:00 | 647.25 | 560.86 | 560.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 790.55 | 791.70 | 748.64 | EMA200 retest candle locked |

### Cycle 8 — SELL (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 10:15:00 | 703.25 | 742.56 | 742.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 700.30 | 742.14 | 742.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 09:15:00 | 729.45 | 722.30 | 729.91 | EMA200 retest candle locked |

### Cycle 9 — BUY (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 13:15:00 | 768.20 | 734.56 | 734.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 14:15:00 | 772.50 | 742.13 | 738.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 11:15:00 | 977.20 | 978.09 | 937.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-16 09:15:00 | 1006.65 | 977.88 | 939.39 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-27 11:15:00 | 950.05 | 984.02 | 950.60 | Close below EMA400 |

### Cycle 10 — SELL (started 2026-03-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 09:15:00 | 889.95 | 961.60 | 961.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 881.70 | 960.81 | 961.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 941.05 | 911.16 | 930.94 | EMA200 retest candle locked |

### Cycle 11 — BUY (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 15:15:00 | 1040.00 | 945.17 | 945.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 09:15:00 | 1041.80 | 946.13 | 945.51 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-12-20 09:15:00 | 758.35 | 2024-01-17 11:15:00 | 747.40 | EXIT_EMA400 | -10.95 |
| SELL | 2024-04-03 10:15:00 | 579.05 | 2024-04-04 09:15:00 | 634.50 | EXIT_EMA400 | -55.45 |
| BUY | 2024-06-05 10:15:00 | 648.15 | 2024-07-08 11:15:00 | 646.35 | EXIT_EMA400 | -1.80 |
| BUY | 2024-06-24 10:15:00 | 671.70 | 2024-07-08 11:15:00 | 646.35 | EXIT_EMA400 | -25.35 |
| SELL | 2024-08-23 11:15:00 | 629.20 | 2024-08-29 11:15:00 | 648.95 | EXIT_EMA400 | -19.75 |
| SELL | 2024-08-28 15:15:00 | 628.30 | 2024-08-29 11:15:00 | 648.95 | EXIT_EMA400 | -20.65 |
| SELL | 2025-02-14 09:15:00 | 545.05 | 2025-03-25 10:15:00 | 556.70 | EXIT_EMA400 | -11.65 |
| SELL | 2025-03-24 11:15:00 | 548.30 | 2025-03-25 10:15:00 | 556.70 | EXIT_EMA400 | -8.40 |
| BUY | 2026-01-16 09:15:00 | 1006.65 | 2026-01-27 11:15:00 | 950.05 | EXIT_EMA400 | -56.60 |
