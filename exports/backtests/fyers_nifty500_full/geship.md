# Great Eastern Shipping Co. Ltd. (GESHIP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1583.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 4 |
| ENTRY1 | 3 |
| ENTRY2 | 3 |
| EXIT | 3 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 5 / 1
- **Target hits / EMA400 exits:** 3 / 3
- **Total realized P&L (per unit):** 271.14
- **Avg P&L per closed trade:** 45.19

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 14:15:00 | 1204.95 | 1272.23 | 1272.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 1190.45 | 1267.00 | 1269.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 12:15:00 | 1251.50 | 1250.01 | 1259.92 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 11:15:00 | 1280.35 | 1267.07 | 1267.02 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-10-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 12:15:00 | 1263.30 | 1266.95 | 1266.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 15:15:00 | 1242.75 | 1266.58 | 1266.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 10:15:00 | 1261.65 | 1253.19 | 1259.20 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 15:15:00 | 1304.00 | 1264.01 | 1264.01 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 09:15:00 | 1208.40 | 1263.46 | 1263.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 13:15:00 | 1183.15 | 1257.16 | 1260.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 10:15:00 | 976.40 | 965.73 | 1022.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-03 09:15:00 | 951.95 | 968.18 | 1019.87 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-19 11:15:00 | 934.95 | 894.94 | 932.78 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 14:15:00 | 931.00 | 917.74 | 917.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 09:15:00 | 965.55 | 918.35 | 918.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 953.10 | 957.65 | 942.26 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 10:15:00 | 961.55 | 957.45 | 942.68 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 961.55 | 957.45 | 942.68 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-20 11:15:00 | 966.70 | 957.54 | 942.80 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 975.75 | 988.21 | 972.96 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-25 09:15:00 | 963.80 | 987.97 | 972.91 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 12:15:00 | 923.00 | 962.31 | 962.44 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 14:15:00 | 983.35 | 961.80 | 961.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 985.00 | 962.24 | 962.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 09:15:00 | 961.95 | 963.66 | 962.77 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 10:15:00 | 942.05 | 961.86 | 961.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 11:15:00 | 938.20 | 961.62 | 961.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 961.35 | 957.52 | 959.61 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 989.45 | 961.49 | 961.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 998.90 | 966.84 | 964.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 11:15:00 | 997.45 | 1000.47 | 984.56 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-03 09:15:00 | 1034.40 | 999.04 | 985.73 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 1016.00 | 1023.88 | 1006.45 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-27 09:15:00 | 1041.35 | 1024.01 | 1007.03 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 1066.40 | 1090.05 | 1066.00 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-08 14:15:00 | 1072.10 | 1089.67 | 1066.05 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-12-09 09:15:00 | 1064.70 | 1089.24 | 1066.06 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-03 09:15:00 | 951.95 | 2025-03-19 11:15:00 | 934.95 | EXIT_EMA400 | 17.00 |
| BUY | 2025-06-20 10:15:00 | 961.55 | 2025-07-03 09:15:00 | 1018.15 | TARGET | 56.60 |
| BUY | 2025-06-20 11:15:00 | 966.70 | 2025-07-04 12:15:00 | 1038.39 | TARGET | 71.69 |
| BUY | 2025-10-27 09:15:00 | 1041.35 | 2025-11-12 09:15:00 | 1144.31 | TARGET | 102.96 |
| BUY | 2025-10-03 09:15:00 | 1034.40 | 2025-12-09 09:15:00 | 1064.70 | EXIT_EMA400 | 30.30 |
| BUY | 2025-12-08 14:15:00 | 1072.10 | 2025-12-09 09:15:00 | 1064.70 | EXIT_EMA400 | -7.40 |
