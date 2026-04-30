# Life Insurance Corporation of India (LICI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 797.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 1 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -136.95
- **Avg P&L per closed trade:** -34.24

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 15:15:00 | 616.55 | 643.37 | 643.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 12:15:00 | 613.55 | 642.30 | 642.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 15:15:00 | 623.10 | 617.85 | 625.98 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 09:15:00 | 683.75 | 633.23 | 633.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 10:15:00 | 697.40 | 636.87 | 634.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 11:15:00 | 1003.15 | 1004.27 | 935.09 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-03-06 14:15:00 | 1009.90 | 1004.35 | 936.16 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-13 10:15:00 | 936.45 | 1004.32 | 943.93 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 14:15:00 | 1021.70 | 1057.41 | 1057.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 15:15:00 | 1020.10 | 1057.04 | 1057.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 953.60 | 946.67 | 976.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-12 13:15:00 | 925.90 | 946.22 | 975.99 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-29 09:15:00 | 969.55 | 926.75 | 955.12 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 14:15:00 | 857.20 | 804.78 | 804.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 10:15:00 | 861.30 | 806.37 | 805.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 934.45 | 935.84 | 905.10 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 15:15:00 | 888.35 | 902.81 | 902.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 877.05 | 902.56 | 902.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 11:15:00 | 888.60 | 885.47 | 891.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-16 13:15:00 | 883.10 | 885.44 | 891.82 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 889.00 | 885.41 | 891.50 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-19 09:15:00 | 894.50 | 885.74 | 891.46 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 14:15:00 | 896.75 | 894.46 | 894.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 09:15:00 | 901.00 | 894.55 | 894.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 10:15:00 | 894.00 | 895.32 | 894.91 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-21 14:15:00 | 892.10 | 894.54 | 894.54 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 903.90 | 894.63 | 894.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 913.55 | 895.03 | 894.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 11:15:00 | 896.65 | 896.66 | 895.69 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-03 10:15:00 | 905.15 | 896.69 | 895.74 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-06 09:15:00 | 896.70 | 898.53 | 896.75 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 14:15:00 | 868.15 | 898.08 | 898.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 863.90 | 894.02 | 895.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 10:15:00 | 842.80 | 831.56 | 849.38 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 13:15:00 | 876.45 | 858.85 | 858.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 14:15:00 | 880.80 | 859.06 | 858.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 857.50 | 861.32 | 860.14 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 822.70 | 858.92 | 858.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 799.20 | 852.70 | 855.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 792.85 | 786.02 | 812.08 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-03-06 14:15:00 | 1009.90 | 2024-03-13 10:15:00 | 936.45 | EXIT_EMA400 | -73.45 |
| SELL | 2024-11-12 13:15:00 | 925.90 | 2024-11-29 09:15:00 | 969.55 | EXIT_EMA400 | -43.65 |
| SELL | 2025-09-16 13:15:00 | 883.10 | 2025-09-19 09:15:00 | 894.50 | EXIT_EMA400 | -11.40 |
| BUY | 2025-11-03 10:15:00 | 905.15 | 2025-11-06 09:15:00 | 896.70 | EXIT_EMA400 | -8.45 |
