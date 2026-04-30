# Adani Total Gas Ltd. (ATGL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 634.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** -81.67
- **Avg P&L per closed trade:** -16.33

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-05 09:15:00 | 767.00 | 602.20 | 602.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 10:15:00 | 804.25 | 604.21 | 603.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 990.95 | 992.70 | 893.97 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-23 09:15:00 | 1026.95 | 992.69 | 900.64 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-12 11:15:00 | 979.45 | 1016.99 | 982.42 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-04-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 12:15:00 | 939.75 | 966.20 | 966.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-18 13:15:00 | 934.85 | 965.89 | 966.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-21 09:15:00 | 926.60 | 921.18 | 936.84 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 11:15:00 | 1117.15 | 946.62 | 946.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 12:15:00 | 1122.80 | 948.38 | 947.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 948.20 | 954.46 | 950.60 | EMA200 retest candle locked |

### Cycle 4 — SELL (started 2024-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 15:15:00 | 923.00 | 949.27 | 949.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 09:15:00 | 919.50 | 948.98 | 949.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 13:15:00 | 909.00 | 903.48 | 917.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-07-29 14:15:00 | 893.65 | 903.38 | 917.50 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 14:15:00 | 893.65 | 903.38 | 917.50 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-08-01 14:15:00 | 916.80 | 902.88 | 915.82 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 14:15:00 | 660.00 | 620.97 | 620.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 15:15:00 | 661.30 | 621.37 | 621.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 662.45 | 667.56 | 651.24 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-27 09:15:00 | 685.80 | 653.98 | 648.26 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-08 10:15:00 | 652.15 | 658.88 | 652.40 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 15:15:00 | 629.50 | 649.96 | 649.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 14:15:00 | 626.10 | 648.77 | 649.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 13:15:00 | 625.20 | 623.02 | 633.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-21 14:15:00 | 618.00 | 623.93 | 632.93 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 631.30 | 623.62 | 632.24 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-08-26 09:15:00 | 636.95 | 623.98 | 632.25 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 13:15:00 | 742.50 | 628.62 | 628.15 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 11:15:00 | 621.00 | 631.06 | 631.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 618.15 | 629.71 | 630.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 629.90 | 625.70 | 627.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-18 10:15:00 | 613.45 | 625.00 | 627.31 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-01 09:15:00 | 613.90 | 586.16 | 599.43 | Close above EMA400 |

### Cycle 9 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 617.10 | 543.73 | 543.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 14:15:00 | 633.00 | 547.49 | 545.37 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-23 09:15:00 | 1026.95 | 2024-03-12 11:15:00 | 979.45 | EXIT_EMA400 | -47.50 |
| SELL | 2024-07-29 14:15:00 | 893.65 | 2024-08-01 14:15:00 | 916.80 | EXIT_EMA400 | -23.15 |
| BUY | 2025-06-27 09:15:00 | 685.80 | 2025-07-08 10:15:00 | 652.15 | EXIT_EMA400 | -33.65 |
| SELL | 2025-08-21 14:15:00 | 618.00 | 2025-08-26 09:15:00 | 636.95 | EXIT_EMA400 | -18.95 |
| SELL | 2025-11-18 10:15:00 | 613.45 | 2025-12-09 09:15:00 | 571.87 | TARGET | 41.58 |
