# Computer Age Management Services Ltd. (CAMS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 738.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** 13.09
- **Avg P&L per closed trade:** 1.87

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 09:15:00 | 460.40 | 484.60 | 484.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-02 14:15:00 | 451.60 | 483.11 | 483.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 09:15:00 | 487.52 | 481.86 | 483.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-07 10:15:00 | 480.72 | 482.01 | 483.26 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-11-08 09:15:00 | 486.29 | 482.01 | 483.22 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 14:15:00 | 529.60 | 484.42 | 484.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-13 15:15:00 | 536.62 | 484.94 | 484.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 09:15:00 | 536.13 | 539.87 | 523.24 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-19 09:15:00 | 545.60 | 539.79 | 523.77 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 09:15:00 | 537.20 | 539.70 | 524.81 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-12-21 14:15:00 | 542.40 | 539.65 | 525.15 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-01-24 09:15:00 | 536.20 | 549.52 | 538.37 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 10:15:00 | 884.60 | 949.71 | 949.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 09:15:00 | 865.98 | 945.76 | 947.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 699.06 | 696.81 | 758.72 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 789.78 | 759.28 | 759.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 14:15:00 | 795.34 | 759.96 | 759.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 794.42 | 803.94 | 785.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 12:15:00 | 807.12 | 803.19 | 786.44 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 812.84 | 830.01 | 812.16 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-11 11:15:00 | 807.66 | 829.79 | 812.14 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 15:15:00 | 758.24 | 809.23 | 809.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 750.46 | 808.64 | 809.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 12:15:00 | 772.68 | 772.05 | 783.79 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-08 13:15:00 | 769.52 | 772.02 | 783.72 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 780.82 | 772.44 | 782.98 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-11 12:15:00 | 776.90 | 772.67 | 782.94 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-17 09:15:00 | 782.00 | 772.70 | 781.73 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 805.00 | 778.07 | 778.03 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 751.50 | 778.46 | 778.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 746.00 | 777.40 | 778.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 11:15:00 | 767.50 | 764.29 | 770.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-23 11:15:00 | 760.90 | 764.19 | 769.96 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-24 09:15:00 | 771.10 | 764.09 | 769.77 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 13:15:00 | 773.00 | 699.88 | 699.74 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-07 10:15:00 | 480.72 | 2023-11-08 09:15:00 | 486.29 | EXIT_EMA400 | -5.57 |
| BUY | 2023-12-19 09:15:00 | 545.60 | 2024-01-24 09:15:00 | 536.20 | EXIT_EMA400 | -9.40 |
| BUY | 2023-12-21 14:15:00 | 542.40 | 2024-01-24 09:15:00 | 536.20 | EXIT_EMA400 | -6.20 |
| BUY | 2025-06-16 12:15:00 | 807.12 | 2025-06-30 09:15:00 | 869.16 | TARGET | 62.04 |
| SELL | 2025-09-08 13:15:00 | 769.52 | 2025-09-17 09:15:00 | 782.00 | EXIT_EMA400 | -12.48 |
| SELL | 2025-09-11 12:15:00 | 776.90 | 2025-09-17 09:15:00 | 782.00 | EXIT_EMA400 | -5.10 |
| SELL | 2025-12-23 11:15:00 | 760.90 | 2025-12-24 09:15:00 | 771.10 | EXIT_EMA400 | -10.20 |
