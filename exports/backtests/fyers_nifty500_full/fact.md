# Fertilisers and Chemicals Travancore Ltd. (FACT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 901.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 1 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / EMA400 exits:** 3 / 3
- **Total realized P&L (per unit):** 116.00
- **Avg P&L per closed trade:** 19.33

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 12:15:00 | 919.15 | 953.13 | 953.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 898.95 | 944.04 | 948.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 14:15:00 | 891.90 | 888.48 | 912.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 11:15:00 | 879.05 | 889.66 | 912.14 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-28 09:15:00 | 906.00 | 863.00 | 888.54 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 11:15:00 | 1028.10 | 908.74 | 908.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 10:15:00 | 1058.65 | 921.83 | 915.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 13:15:00 | 951.85 | 960.86 | 939.44 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-01 09:15:00 | 989.00 | 959.50 | 942.84 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-06 10:15:00 | 944.10 | 964.68 | 947.34 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 11:15:00 | 895.75 | 939.73 | 939.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 14:15:00 | 889.65 | 936.48 | 938.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 10:15:00 | 982.20 | 935.25 | 937.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-01 12:15:00 | 913.75 | 935.10 | 937.36 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 913.75 | 935.10 | 937.36 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-03 09:15:00 | 908.65 | 934.40 | 936.97 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-04-17 10:15:00 | 730.80 | 660.82 | 718.01 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 11:15:00 | 830.00 | 745.50 | 745.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 14:15:00 | 834.00 | 747.97 | 746.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 14:15:00 | 967.40 | 971.67 | 906.26 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-27 09:15:00 | 973.50 | 971.61 | 906.89 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-11 10:15:00 | 916.70 | 957.71 | 918.55 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 14:15:00 | 898.00 | 955.29 | 955.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 15:15:00 | 897.55 | 954.71 | 955.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 14:15:00 | 903.00 | 902.78 | 918.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-21 09:15:00 | 886.80 | 904.18 | 916.42 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-16 09:15:00 | 909.95 | 861.26 | 884.78 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 15:15:00 | 870.00 | 807.96 | 807.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 931.50 | 809.19 | 808.50 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-08 11:15:00 | 879.05 | 2024-11-28 09:15:00 | 906.00 | EXIT_EMA400 | -26.95 |
| BUY | 2025-01-01 09:15:00 | 989.00 | 2025-01-06 10:15:00 | 944.10 | EXIT_EMA400 | -44.90 |
| SELL | 2025-02-01 12:15:00 | 913.75 | 2025-02-11 09:15:00 | 842.91 | TARGET | 70.84 |
| SELL | 2025-02-03 09:15:00 | 908.65 | 2025-02-11 09:15:00 | 823.69 | TARGET | 84.96 |
| BUY | 2025-06-27 09:15:00 | 973.50 | 2025-07-11 10:15:00 | 916.70 | EXIT_EMA400 | -56.80 |
| SELL | 2025-11-21 09:15:00 | 886.80 | 2025-12-08 13:15:00 | 797.95 | TARGET | 88.85 |
