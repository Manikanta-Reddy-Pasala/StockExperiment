# K.P.R. Mill Ltd. (KPRMILL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 936.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -298.80
- **Avg P&L per closed trade:** -74.70

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 12:15:00 | 869.85 | 972.29 | 972.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 09:15:00 | 846.45 | 937.65 | 952.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-06 09:15:00 | 872.25 | 862.36 | 899.64 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2025-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 14:15:00 | 926.95 | 909.56 | 909.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 10:15:00 | 964.65 | 911.19 | 910.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 09:15:00 | 1110.00 | 1116.81 | 1057.67 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-04 15:15:00 | 1138.90 | 1118.77 | 1064.31 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1078.30 | 1118.92 | 1076.68 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-16 10:15:00 | 1076.20 | 1118.49 | 1076.68 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 12:15:00 | 987.50 | 1116.41 | 1116.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 09:15:00 | 975.60 | 1111.38 | 1114.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 1023.20 | 1022.37 | 1051.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-09 10:15:00 | 1009.20 | 1055.91 | 1061.60 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 1043.00 | 1041.24 | 1052.32 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-20 09:15:00 | 1021.00 | 1041.10 | 1051.87 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-23 09:15:00 | 1075.00 | 1040.28 | 1050.97 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 09:15:00 | 1080.30 | 1057.11 | 1057.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 10:15:00 | 1092.50 | 1060.73 | 1058.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 1063.90 | 1074.62 | 1067.61 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 14:15:00 | 984.60 | 1061.45 | 1061.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 15:15:00 | 978.10 | 1060.62 | 1061.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 12:15:00 | 911.50 | 897.87 | 945.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-29 11:15:00 | 885.00 | 899.53 | 943.73 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-03 09:15:00 | 1001.30 | 894.28 | 935.52 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 14:15:00 | 925.00 | 891.32 | 891.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 10:15:00 | 934.85 | 892.46 | 891.89 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-06-04 15:15:00 | 1138.90 | 2025-06-16 10:15:00 | 1076.20 | EXIT_EMA400 | -62.70 |
| SELL | 2025-10-09 10:15:00 | 1009.20 | 2025-10-23 09:15:00 | 1075.00 | EXIT_EMA400 | -65.80 |
| SELL | 2025-10-20 09:15:00 | 1021.00 | 2025-10-23 09:15:00 | 1075.00 | EXIT_EMA400 | -54.00 |
| SELL | 2026-01-29 11:15:00 | 885.00 | 2026-02-03 09:15:00 | 1001.30 | EXIT_EMA400 | -116.30 |
