# Jindal Steel Ltd. (JINDALSTEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1223.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 6 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 98.73
- **Avg P&L per closed trade:** 16.45

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 13:15:00 | 590.90 | 661.25 | 661.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-02 14:15:00 | 588.75 | 660.52 | 660.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 09:15:00 | 652.35 | 646.26 | 652.57 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 13:15:00 | 681.90 | 655.54 | 655.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-06 11:15:00 | 683.80 | 658.18 | 656.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 14:15:00 | 722.50 | 722.59 | 704.17 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-19 09:15:00 | 728.20 | 722.31 | 704.83 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-01-23 13:15:00 | 699.35 | 722.09 | 705.66 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 11:15:00 | 938.40 | 982.46 | 982.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 09:15:00 | 927.90 | 980.40 | 981.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 14:15:00 | 958.20 | 958.01 | 968.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-22 09:15:00 | 951.75 | 957.92 | 967.89 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 963.15 | 958.10 | 967.53 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-08-23 12:15:00 | 968.00 | 958.19 | 967.54 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 14:15:00 | 1042.05 | 969.85 | 969.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 10:15:00 | 1050.95 | 972.05 | 970.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 10:15:00 | 999.30 | 1007.44 | 992.84 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-10-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 15:15:00 | 910.50 | 984.99 | 985.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 898.00 | 976.01 | 980.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 09:15:00 | 960.75 | 952.88 | 966.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-07 12:15:00 | 939.70 | 952.73 | 965.74 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 926.75 | 913.29 | 935.25 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-04 14:15:00 | 935.25 | 914.96 | 934.83 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 14:15:00 | 922.80 | 890.77 | 890.68 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 11:15:00 | 790.35 | 891.61 | 891.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 12:15:00 | 786.45 | 890.57 | 891.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-17 10:15:00 | 887.20 | 867.02 | 877.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-08 15:15:00 | 851.15 | 882.85 | 884.05 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 881.40 | 880.66 | 882.88 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-05-12 10:15:00 | 883.55 | 880.69 | 882.89 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 12:15:00 | 928.25 | 885.09 | 884.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 13:15:00 | 948.70 | 885.72 | 885.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 937.20 | 942.36 | 923.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-26 14:15:00 | 954.20 | 928.07 | 920.80 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 936.60 | 939.37 | 929.16 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-14 12:15:00 | 928.15 | 939.47 | 930.29 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 1012.70 | 1031.48 | 1031.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 1007.80 | 1031.04 | 1031.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 1021.30 | 1016.61 | 1022.90 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 13:15:00 | 1077.80 | 1028.61 | 1028.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 14:15:00 | 1082.30 | 1029.15 | 1028.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 1034.40 | 1036.10 | 1032.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-22 14:15:00 | 1075.20 | 1037.50 | 1033.81 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-09 10:15:00 | 1136.10 | 1179.69 | 1136.40 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-19 09:15:00 | 728.20 | 2024-01-23 13:15:00 | 699.35 | EXIT_EMA400 | -28.85 |
| SELL | 2024-08-22 09:15:00 | 951.75 | 2024-08-23 12:15:00 | 968.00 | EXIT_EMA400 | -16.25 |
| SELL | 2024-11-07 12:15:00 | 939.70 | 2024-11-13 11:15:00 | 861.59 | TARGET | 78.11 |
| SELL | 2025-05-08 15:15:00 | 851.15 | 2025-05-12 10:15:00 | 883.55 | EXIT_EMA400 | -32.40 |
| BUY | 2025-06-26 14:15:00 | 954.20 | 2025-07-14 12:15:00 | 928.15 | EXIT_EMA400 | -26.05 |
| BUY | 2026-01-22 14:15:00 | 1075.20 | 2026-02-09 11:15:00 | 1199.36 | TARGET | 124.16 |
