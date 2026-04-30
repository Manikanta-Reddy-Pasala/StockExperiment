# Fertilisers and Chemicals Travancore Ltd. (FACT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 899.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 3 |
| EXIT | 6 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 0 / 9
- **Target hits / EMA400 exits:** 0 / 9
- **Total realized P&L (per unit):** -317.75
- **Avg P&L per closed trade:** -35.31

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-07 11:15:00 | 716.55 | 771.37 | 771.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-07 14:15:00 | 714.75 | 769.73 | 770.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-03 10:15:00 | 698.05 | 691.30 | 720.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-09 13:15:00 | 685.10 | 692.16 | 716.83 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-24 09:15:00 | 709.50 | 680.98 | 704.23 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 11:15:00 | 779.05 | 702.12 | 701.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 09:15:00 | 787.00 | 705.71 | 703.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 09:15:00 | 999.90 | 1002.87 | 929.07 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-22 09:15:00 | 1036.90 | 974.24 | 938.98 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 960.00 | 983.47 | 956.70 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-09-09 15:15:00 | 989.00 | 982.74 | 957.12 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 973.60 | 986.22 | 965.03 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-09-23 10:15:00 | 990.95 | 985.07 | 965.77 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-09-26 13:15:00 | 965.70 | 983.83 | 967.32 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 15:15:00 | 884.95 | 956.34 | 956.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 14:15:00 | 880.00 | 938.98 | 946.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 14:15:00 | 891.05 | 889.06 | 914.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 11:15:00 | 879.05 | 890.21 | 913.52 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-28 09:15:00 | 906.00 | 863.13 | 889.39 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 12:15:00 | 1025.95 | 910.00 | 909.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 14:15:00 | 1028.60 | 912.34 | 910.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 13:15:00 | 951.85 | 960.91 | 939.93 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-01 09:15:00 | 989.00 | 959.56 | 943.24 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-06 10:15:00 | 944.55 | 964.74 | 947.71 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 10:15:00 | 899.65 | 940.27 | 940.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-29 11:15:00 | 895.75 | 939.83 | 940.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 12:15:00 | 667.00 | 658.74 | 720.63 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 11:15:00 | 830.00 | 745.60 | 745.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 14:15:00 | 834.00 | 748.06 | 746.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 13:15:00 | 971.80 | 971.89 | 906.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-27 10:15:00 | 975.15 | 971.83 | 907.39 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-11 10:15:00 | 916.70 | 957.84 | 918.69 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 13:15:00 | 899.50 | 955.85 | 955.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 14:15:00 | 898.30 | 955.27 | 955.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 10:15:00 | 914.25 | 914.24 | 930.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-28 15:15:00 | 898.95 | 913.78 | 929.49 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 903.65 | 912.27 | 927.21 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-31 15:15:00 | 897.30 | 912.12 | 927.06 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 913.90 | 902.90 | 918.05 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-17 09:15:00 | 920.00 | 903.45 | 917.88 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 931.50 | 809.30 | 808.87 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-04-09 13:15:00 | 685.10 | 2024-04-24 09:15:00 | 709.50 | EXIT_EMA400 | -24.40 |
| BUY | 2024-08-22 09:15:00 | 1036.90 | 2024-09-26 13:15:00 | 965.70 | EXIT_EMA400 | -71.20 |
| BUY | 2024-09-09 15:15:00 | 989.00 | 2024-09-26 13:15:00 | 965.70 | EXIT_EMA400 | -23.30 |
| BUY | 2024-09-23 10:15:00 | 990.95 | 2024-09-26 13:15:00 | 965.70 | EXIT_EMA400 | -25.25 |
| SELL | 2024-11-08 11:15:00 | 879.05 | 2024-11-28 09:15:00 | 906.00 | EXIT_EMA400 | -26.95 |
| BUY | 2025-01-01 09:15:00 | 989.00 | 2025-01-06 10:15:00 | 944.55 | EXIT_EMA400 | -44.45 |
| BUY | 2025-06-27 10:15:00 | 975.15 | 2025-07-11 10:15:00 | 916.70 | EXIT_EMA400 | -58.45 |
| SELL | 2025-10-28 15:15:00 | 898.95 | 2025-11-17 09:15:00 | 920.00 | EXIT_EMA400 | -21.05 |
| SELL | 2025-10-31 15:15:00 | 897.30 | 2025-11-17 09:15:00 | 920.00 | EXIT_EMA400 | -22.70 |
