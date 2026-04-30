# Jindal Steel Ltd. (JINDALSTEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1229.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 2 / 1
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 90.61
- **Avg P&L per closed trade:** 30.20

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 14:15:00 | 1050.20 | 975.38 | 975.28 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-10-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 12:15:00 | 922.65 | 987.18 | 987.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 13:15:00 | 919.65 | 986.51 | 987.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 09:15:00 | 960.75 | 952.47 | 966.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-07 12:15:00 | 939.70 | 952.33 | 966.19 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 926.75 | 913.07 | 935.45 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-04 14:15:00 | 935.25 | 914.77 | 935.02 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 10:15:00 | 923.00 | 888.66 | 888.59 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-04-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 13:15:00 | 796.35 | 889.17 | 889.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 09:15:00 | 770.25 | 880.06 | 884.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-17 10:15:00 | 887.20 | 866.64 | 876.89 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2025-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 12:15:00 | 902.30 | 884.38 | 884.31 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 11:15:00 | 879.75 | 884.26 | 884.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 10:15:00 | 870.10 | 883.83 | 884.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 881.40 | 880.43 | 882.28 | EMA200 retest candle locked |

### Cycle 7 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 927.75 | 884.09 | 884.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 13:15:00 | 948.70 | 885.54 | 884.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 937.20 | 942.23 | 923.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-26 14:15:00 | 954.20 | 927.94 | 920.54 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 936.60 | 939.18 | 928.91 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-14 12:15:00 | 928.15 | 939.31 | 930.06 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 1012.70 | 1031.40 | 1031.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 1007.80 | 1030.95 | 1031.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 1021.30 | 1016.55 | 1022.83 | EMA200 retest candle locked |

### Cycle 9 — BUY (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 13:15:00 | 1077.80 | 1028.55 | 1028.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 14:15:00 | 1082.30 | 1029.09 | 1028.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 1034.40 | 1036.05 | 1032.30 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-23 09:15:00 | 1098.50 | 1038.48 | 1034.31 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-09 10:15:00 | 1135.70 | 1180.42 | 1137.47 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-07 12:15:00 | 939.70 | 2024-11-13 11:15:00 | 860.24 | TARGET | 79.46 |
| BUY | 2025-06-26 14:15:00 | 954.20 | 2025-07-14 12:15:00 | 928.15 | EXIT_EMA400 | -26.05 |
| BUY | 2026-01-23 09:15:00 | 1098.50 | 2026-03-09 10:15:00 | 1135.70 | EXIT_EMA400 | 37.20 |
