# Newgen Software Technologies Ltd. (NEWGEN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 508.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -194.30
- **Avg P&L per closed trade:** -32.38

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 1059.85 | 1383.03 | 1383.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 1018.60 | 1260.15 | 1314.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 09:15:00 | 1023.20 | 1004.20 | 1099.13 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-04 09:15:00 | 949.15 | 1002.58 | 1072.88 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-24 09:15:00 | 1064.20 | 961.91 | 1024.96 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 1166.50 | 1053.74 | 1053.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 1218.00 | 1060.17 | 1056.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 1173.80 | 1182.30 | 1139.02 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 12:15:00 | 1240.10 | 1181.62 | 1141.38 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 1152.70 | 1187.51 | 1150.26 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-23 13:15:00 | 1146.40 | 1185.46 | 1150.32 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 12:15:00 | 1027.90 | 1137.35 | 1137.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 972.70 | 1132.42 | 1135.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 933.45 | 923.06 | 985.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-26 09:15:00 | 899.10 | 922.59 | 983.63 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 932.65 | 900.51 | 943.62 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-23 10:15:00 | 883.50 | 902.38 | 940.16 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 917.50 | 900.06 | 936.58 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-25 13:15:00 | 902.85 | 900.51 | 936.09 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-23 09:15:00 | 912.80 | 885.74 | 911.10 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 959.30 | 925.94 | 925.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 971.00 | 926.71 | 926.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 09:15:00 | 929.90 | 933.30 | 929.88 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 13:15:00 | 891.55 | 926.99 | 927.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 09:15:00 | 877.40 | 925.73 | 926.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-19 10:15:00 | 610.75 | 605.51 | 692.88 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-20 09:15:00 | 582.95 | 605.81 | 690.45 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-30 12:15:00 | 515.55 | 473.68 | 512.86 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-04 09:15:00 | 949.15 | 2025-04-24 09:15:00 | 1064.20 | EXIT_EMA400 | -115.05 |
| BUY | 2025-06-16 12:15:00 | 1240.10 | 2025-06-23 13:15:00 | 1146.40 | EXIT_EMA400 | -93.70 |
| SELL | 2025-08-26 09:15:00 | 899.10 | 2025-10-23 09:15:00 | 912.80 | EXIT_EMA400 | -13.70 |
| SELL | 2025-09-23 10:15:00 | 883.50 | 2025-10-23 09:15:00 | 912.80 | EXIT_EMA400 | -29.30 |
| SELL | 2025-09-25 13:15:00 | 902.85 | 2025-10-23 09:15:00 | 912.80 | EXIT_EMA400 | -9.95 |
| SELL | 2026-02-20 09:15:00 | 582.95 | 2026-04-30 12:15:00 | 515.55 | EXIT_EMA400 | 67.40 |
