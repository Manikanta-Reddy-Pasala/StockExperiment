# E.I.D. Parry (India) Ltd. (EIDPARRY.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 844.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 8 |
| ENTRY1 | 6 |
| ENTRY2 | 7 |
| EXIT | 6 |

## P&L

- **Trades closed:** 13
- **Trades open at end:** 0
- **Winners / losers:** 6 / 7
- **Target hits / EMA400 exits:** 5 / 8
- **Total realized P&L (per unit):** 248.44
- **Avg P&L per closed trade:** 19.11

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 15:15:00 | 480.70 | 474.37 | 474.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 09:15:00 | 484.05 | 474.46 | 474.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 11:15:00 | 511.25 | 515.28 | 501.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-10-05 09:15:00 | 517.85 | 515.15 | 501.64 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 10:15:00 | 505.65 | 516.83 | 505.22 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-10-13 11:15:00 | 509.10 | 516.75 | 505.24 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 11:15:00 | 508.10 | 515.85 | 505.91 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-10-18 13:15:00 | 508.60 | 515.69 | 505.94 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2023-10-18 14:15:00 | 496.80 | 515.50 | 505.89 | Close below EMA400 |

### Cycle 2 — SELL (started 2023-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 12:15:00 | 466.30 | 498.76 | 498.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 14:15:00 | 462.05 | 498.08 | 498.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 14:15:00 | 492.40 | 490.38 | 494.10 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2023-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 11:15:00 | 536.35 | 496.53 | 496.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-24 13:15:00 | 539.70 | 497.30 | 496.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-09 10:15:00 | 586.55 | 596.01 | 571.89 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-15 10:15:00 | 598.80 | 593.70 | 573.79 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 09:15:00 | 594.35 | 611.26 | 594.25 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-03-12 10:15:00 | 587.00 | 611.02 | 594.22 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 15:15:00 | 543.20 | 582.79 | 582.92 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 13:15:00 | 593.00 | 582.65 | 582.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-09 14:15:00 | 604.00 | 582.86 | 582.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 11:15:00 | 606.35 | 608.32 | 599.02 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-07 14:15:00 | 609.95 | 608.33 | 599.17 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 609.00 | 609.47 | 600.47 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-05-14 09:15:00 | 621.50 | 609.48 | 601.07 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 757.35 | 775.94 | 739.54 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-08-06 09:15:00 | 772.45 | 774.43 | 740.02 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 14:15:00 | 751.00 | 774.14 | 746.99 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-08-19 09:15:00 | 787.95 | 772.90 | 747.55 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 796.45 | 812.23 | 788.43 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-09-19 12:15:00 | 805.05 | 812.15 | 788.51 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-10-07 10:15:00 | 794.50 | 827.51 | 804.70 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 10:15:00 | 796.45 | 857.09 | 857.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 11:15:00 | 785.15 | 856.37 | 856.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 11:15:00 | 843.30 | 842.07 | 848.79 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-05 10:15:00 | 835.30 | 841.96 | 848.54 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-06 09:15:00 | 858.45 | 841.87 | 848.30 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 10:15:00 | 859.65 | 778.61 | 778.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 10:15:00 | 868.40 | 788.05 | 783.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 09:15:00 | 946.65 | 949.92 | 905.39 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-18 10:15:00 | 950.00 | 949.92 | 905.61 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 1116.30 | 1151.95 | 1106.61 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-01 09:15:00 | 1128.90 | 1150.17 | 1107.27 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-09-02 13:15:00 | 1106.50 | 1147.63 | 1108.26 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 15:15:00 | 1015.10 | 1093.05 | 1093.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 09:15:00 | 1008.00 | 1054.79 | 1065.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 1071.90 | 1053.06 | 1064.00 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-12 11:15:00 | 1039.70 | 1052.94 | 1063.83 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-11-13 09:15:00 | 1069.00 | 1052.76 | 1063.47 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-10-05 09:15:00 | 517.85 | 2023-10-18 14:15:00 | 496.80 | EXIT_EMA400 | -21.05 |
| BUY | 2023-10-13 11:15:00 | 509.10 | 2023-10-18 14:15:00 | 496.80 | EXIT_EMA400 | -12.30 |
| BUY | 2023-10-18 13:15:00 | 508.60 | 2023-10-18 14:15:00 | 496.80 | EXIT_EMA400 | -11.80 |
| BUY | 2024-02-15 10:15:00 | 598.80 | 2024-03-12 10:15:00 | 587.00 | EXIT_EMA400 | -11.80 |
| BUY | 2024-05-07 14:15:00 | 609.95 | 2024-05-22 09:15:00 | 642.30 | TARGET | 32.35 |
| BUY | 2024-05-14 09:15:00 | 621.50 | 2024-05-29 11:15:00 | 682.78 | TARGET | 61.28 |
| BUY | 2024-08-06 09:15:00 | 772.45 | 2024-08-30 09:15:00 | 869.73 | TARGET | 97.28 |
| BUY | 2024-09-19 12:15:00 | 805.05 | 2024-09-27 09:15:00 | 854.66 | TARGET | 49.61 |
| BUY | 2024-08-19 09:15:00 | 787.95 | 2024-10-07 10:15:00 | 794.50 | EXIT_EMA400 | 6.55 |
| SELL | 2025-02-05 10:15:00 | 835.30 | 2025-02-06 09:15:00 | 858.45 | EXIT_EMA400 | -23.15 |
| BUY | 2025-06-18 10:15:00 | 950.00 | 2025-06-30 11:15:00 | 1083.17 | TARGET | 133.17 |
| BUY | 2025-09-01 09:15:00 | 1128.90 | 2025-09-02 13:15:00 | 1106.50 | EXIT_EMA400 | -22.40 |
| SELL | 2025-11-12 11:15:00 | 1039.70 | 2025-11-13 09:15:00 | 1069.00 | EXIT_EMA400 | -29.30 |
