# Newgen Software Technologies Ltd. (NEWGEN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 505.75
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
- **Total realized P&L (per unit):** -191.50
- **Avg P&L per closed trade:** -31.92

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 1059.85 | 1382.95 | 1383.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 14:15:00 | 1047.25 | 1379.61 | 1381.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 09:15:00 | 1023.10 | 1006.06 | 1102.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-04 09:15:00 | 949.15 | 1003.73 | 1075.19 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-24 09:15:00 | 1064.20 | 962.56 | 1026.61 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 12:15:00 | 1157.85 | 1056.06 | 1055.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 1217.25 | 1060.45 | 1057.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 1173.90 | 1182.41 | 1139.56 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 12:15:00 | 1240.10 | 1181.67 | 1141.87 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 1152.80 | 1187.56 | 1150.69 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-23 12:15:00 | 1150.40 | 1185.91 | 1150.76 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 12:15:00 | 1027.90 | 1137.30 | 1137.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 972.70 | 1132.37 | 1135.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 932.80 | 923.04 | 985.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-26 09:15:00 | 899.10 | 922.57 | 983.69 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 931.85 | 900.49 | 943.64 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-23 10:15:00 | 883.50 | 902.35 | 940.18 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 918.40 | 900.05 | 936.60 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-25 13:15:00 | 902.85 | 900.50 | 936.11 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-23 09:15:00 | 913.20 | 885.80 | 911.16 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 959.25 | 926.07 | 926.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 971.00 | 926.83 | 926.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 09:15:00 | 929.90 | 933.41 | 929.98 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 13:15:00 | 891.55 | 927.07 | 927.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 09:15:00 | 877.40 | 925.84 | 926.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-19 10:15:00 | 610.65 | 609.64 | 697.57 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-20 09:15:00 | 582.95 | 609.70 | 695.00 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-30 12:15:00 | 515.55 | 473.70 | 514.58 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-04 09:15:00 | 949.15 | 2025-04-24 09:15:00 | 1064.20 | EXIT_EMA400 | -115.05 |
| BUY | 2025-06-16 12:15:00 | 1240.10 | 2025-06-23 12:15:00 | 1150.40 | EXIT_EMA400 | -89.70 |
| SELL | 2025-08-26 09:15:00 | 899.10 | 2025-10-23 09:15:00 | 913.20 | EXIT_EMA400 | -14.10 |
| SELL | 2025-09-23 10:15:00 | 883.50 | 2025-10-23 09:15:00 | 913.20 | EXIT_EMA400 | -29.70 |
| SELL | 2025-09-25 13:15:00 | 902.85 | 2025-10-23 09:15:00 | 913.20 | EXIT_EMA400 | -10.35 |
| SELL | 2026-02-20 09:15:00 | 582.95 | 2026-04-30 12:15:00 | 515.55 | EXIT_EMA400 | 67.40 |
