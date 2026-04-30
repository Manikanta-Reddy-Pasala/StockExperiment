# Schneider Electric Infrastructure Ltd. (SCHNEIDER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 1243.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / EMA400 exits:** 3 / 2
- **Total realized P&L (per unit):** 373.33
- **Avg P&L per closed trade:** 74.67

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 12:15:00 | 768.15 | 807.35 | 807.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 13:15:00 | 766.65 | 806.94 | 807.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-24 09:15:00 | 800.00 | 799.18 | 803.05 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 15:15:00 | 827.80 | 806.78 | 806.68 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 757.00 | 806.73 | 806.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 13:15:00 | 744.70 | 805.13 | 806.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 10:15:00 | 805.25 | 802.16 | 804.41 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 15:15:00 | 862.00 | 806.90 | 806.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 13:15:00 | 868.70 | 816.00 | 811.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 10:15:00 | 817.30 | 821.57 | 815.04 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 732.25 | 809.22 | 809.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 10:15:00 | 731.40 | 808.45 | 808.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 806.65 | 789.12 | 797.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 13:15:00 | 778.65 | 790.43 | 797.76 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 15:15:00 | 779.15 | 770.76 | 784.70 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-25 09:15:00 | 809.85 | 771.15 | 784.83 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 15:15:00 | 828.60 | 794.64 | 794.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 13:15:00 | 834.90 | 796.30 | 795.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 09:15:00 | 812.35 | 813.11 | 805.39 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2024-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 15:15:00 | 766.65 | 799.41 | 799.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 758.95 | 797.00 | 798.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 14:15:00 | 643.60 | 637.66 | 671.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-04 09:15:00 | 631.00 | 652.96 | 670.32 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 14:15:00 | 608.80 | 603.03 | 627.95 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-05-14 09:15:00 | 628.65 | 604.31 | 627.51 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 14:15:00 | 781.65 | 642.69 | 642.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 11:15:00 | 792.10 | 687.39 | 666.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 09:15:00 | 888.00 | 910.71 | 845.74 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-08 09:15:00 | 902.95 | 876.92 | 853.84 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-24 10:15:00 | 861.30 | 882.88 | 865.29 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 12:15:00 | 813.00 | 854.71 | 854.87 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 14:15:00 | 877.65 | 852.19 | 852.16 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 10:15:00 | 816.95 | 851.90 | 852.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 15:15:00 | 805.95 | 849.75 | 850.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 751.85 | 746.76 | 778.51 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-26 10:15:00 | 732.05 | 745.83 | 774.72 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-03 09:15:00 | 709.95 | 673.90 | 708.99 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 830.90 | 731.34 | 731.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 13:15:00 | 844.80 | 732.47 | 731.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 851.45 | 858.54 | 818.87 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-24 14:15:00 | 897.55 | 859.74 | 821.79 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-08 13:15:00 | 778.65 | 2024-11-13 14:15:00 | 721.31 | TARGET | 57.34 |
| SELL | 2025-04-04 09:15:00 | 631.00 | 2025-05-14 09:15:00 | 628.65 | EXIT_EMA400 | 2.35 |
| BUY | 2025-09-08 09:15:00 | 902.95 | 2025-09-24 10:15:00 | 861.30 | EXIT_EMA400 | -41.65 |
| SELL | 2025-12-26 10:15:00 | 732.05 | 2026-01-20 09:15:00 | 604.04 | TARGET | 128.01 |
| BUY | 2026-03-24 14:15:00 | 897.55 | 2026-04-23 09:15:00 | 1124.83 | TARGET | 227.28 |
