# Adani Energy Solutions Ltd. (ADANIENSOL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1344.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 4 / 4
- **Target hits / EMA400 exits:** 4 / 4
- **Total realized P&L (per unit):** 245.34
- **Avg P&L per closed trade:** 30.67

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 15:15:00 | 986.45 | 1052.46 | 1052.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 10:15:00 | 976.05 | 1051.11 | 1051.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 09:15:00 | 1024.15 | 1021.79 | 1034.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-27 14:15:00 | 1004.35 | 1026.94 | 1035.45 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 1004.35 | 1026.94 | 1035.45 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-10-01 09:15:00 | 1044.75 | 1025.89 | 1034.54 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 874.40 | 769.68 | 769.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 10:15:00 | 885.25 | 772.91 | 771.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 786.70 | 800.03 | 786.26 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 10:15:00 | 816.60 | 800.19 | 786.41 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 10:15:00 | 816.60 | 800.19 | 786.41 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-04-07 15:15:00 | 821.00 | 800.60 | 786.96 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-05-08 14:15:00 | 842.30 | 877.88 | 843.90 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 09:15:00 | 807.50 | 865.25 | 865.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 10:15:00 | 803.90 | 864.64 | 864.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 10:15:00 | 825.50 | 823.01 | 839.28 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-21 12:15:00 | 816.40 | 823.18 | 838.12 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 813.40 | 795.67 | 816.10 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-10 13:15:00 | 810.70 | 795.82 | 816.08 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-11 09:15:00 | 824.55 | 796.40 | 816.07 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 09:15:00 | 893.35 | 828.53 | 828.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 14:15:00 | 926.20 | 855.34 | 843.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 15:15:00 | 968.00 | 969.13 | 932.39 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-25 09:15:00 | 974.00 | 969.18 | 932.59 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 956.75 | 974.31 | 946.64 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-09 10:15:00 | 967.25 | 974.24 | 946.74 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 987.40 | 1008.26 | 982.84 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-09 11:15:00 | 972.80 | 1007.66 | 982.79 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 891.60 | 965.36 | 965.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 866.80 | 964.38 | 964.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 944.40 | 930.80 | 945.89 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 10:15:00 | 1029.55 | 957.75 | 957.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 12:15:00 | 1031.85 | 959.20 | 958.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 971.80 | 990.61 | 977.73 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-05 15:15:00 | 989.70 | 986.72 | 976.90 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-09 09:15:00 | 971.10 | 987.17 | 977.52 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-27 14:15:00 | 1004.35 | 2024-10-01 09:15:00 | 1044.75 | EXIT_EMA400 | -40.40 |
| BUY | 2025-04-07 10:15:00 | 816.60 | 2025-04-17 09:15:00 | 907.17 | TARGET | 90.57 |
| BUY | 2025-04-07 15:15:00 | 821.00 | 2025-04-17 10:15:00 | 923.13 | TARGET | 102.13 |
| SELL | 2025-08-21 12:15:00 | 816.40 | 2025-09-05 10:15:00 | 751.23 | TARGET | 65.17 |
| SELL | 2025-09-10 13:15:00 | 810.70 | 2025-09-11 09:15:00 | 824.55 | EXIT_EMA400 | -13.85 |
| BUY | 2025-12-09 10:15:00 | 967.25 | 2025-12-31 14:15:00 | 1028.78 | TARGET | 61.53 |
| BUY | 2025-11-25 09:15:00 | 974.00 | 2026-01-09 11:15:00 | 972.80 | EXIT_EMA400 | -1.20 |
| BUY | 2026-03-05 15:15:00 | 989.70 | 2026-03-09 09:15:00 | 971.10 | EXIT_EMA400 | -18.60 |
