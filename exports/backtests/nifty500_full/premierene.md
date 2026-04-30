# Premier Energies Ltd. (PREMIERENE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-09-03 09:15:00 → 2026-04-30 15:30:00 (2848 bars)
- **Last close:** 1018.15
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 7 |
| ENTRY1 | 5 |
| ENTRY2 | 5 |
| EXIT | 5 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 2 / 8
- **Target hits / EMA400 exits:** 2 / 8
- **Total realized P&L (per unit):** -115.80
- **Avg P&L per closed trade:** -11.58

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 15:15:00 | 933.70 | 1168.71 | 1168.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 855.80 | 1165.60 | 1167.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 14:15:00 | 946.75 | 946.65 | 1002.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-21 09:15:00 | 922.70 | 946.40 | 1002.17 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 951.00 | 910.92 | 956.23 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-17 11:15:00 | 961.25 | 911.85 | 956.25 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 13:15:00 | 1122.00 | 978.63 | 978.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 15:15:00 | 1135.00 | 981.59 | 979.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 1026.40 | 1042.25 | 1020.32 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-27 09:15:00 | 1051.90 | 1028.55 | 1018.96 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 1051.90 | 1058.62 | 1041.98 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-23 11:15:00 | 1065.00 | 1057.79 | 1042.44 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 1048.00 | 1062.97 | 1046.88 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-29 10:15:00 | 1061.30 | 1062.70 | 1047.06 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 1055.30 | 1062.52 | 1047.13 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-30 09:15:00 | 1081.80 | 1062.37 | 1047.36 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-31 09:15:00 | 1043.10 | 1062.75 | 1048.07 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 1013.50 | 1037.53 | 1037.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 09:15:00 | 994.00 | 1036.08 | 1036.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 1030.00 | 1028.91 | 1032.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-21 13:15:00 | 1023.00 | 1029.64 | 1033.03 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 1023.00 | 1029.64 | 1033.03 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-22 10:15:00 | 1021.20 | 1029.41 | 1032.85 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-01 12:15:00 | 1031.15 | 1022.31 | 1028.37 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 12:15:00 | 1076.40 | 1030.55 | 1030.47 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 09:15:00 | 1018.80 | 1031.15 | 1031.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 1013.70 | 1028.98 | 1030.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 1027.90 | 1027.80 | 1029.38 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 13:15:00 | 1071.00 | 1030.68 | 1030.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 1097.20 | 1041.27 | 1036.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 10:15:00 | 1053.00 | 1055.96 | 1045.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-12 14:15:00 | 1064.90 | 1045.28 | 1041.53 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-14 09:15:00 | 1032.50 | 1045.83 | 1041.97 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 997.80 | 1038.25 | 1038.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 986.40 | 1037.74 | 1038.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 799.00 | 774.74 | 841.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-04 09:15:00 | 774.85 | 775.96 | 839.55 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 789.00 | 782.08 | 832.57 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-11 11:15:00 | 783.65 | 782.13 | 832.09 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 776.55 | 750.05 | 787.82 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-03-11 14:15:00 | 790.40 | 750.71 | 787.78 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 11:15:00 | 890.35 | 809.01 | 808.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-30 15:15:00 | 898.70 | 812.15 | 810.40 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-21 09:15:00 | 922.70 | 2025-04-17 11:15:00 | 961.25 | EXIT_EMA400 | -38.55 |
| BUY | 2025-06-27 09:15:00 | 1051.90 | 2025-07-31 09:15:00 | 1043.10 | EXIT_EMA400 | -8.80 |
| BUY | 2025-07-23 11:15:00 | 1065.00 | 2025-07-31 09:15:00 | 1043.10 | EXIT_EMA400 | -21.90 |
| BUY | 2025-07-29 10:15:00 | 1061.30 | 2025-07-31 09:15:00 | 1043.10 | EXIT_EMA400 | -18.20 |
| BUY | 2025-07-30 09:15:00 | 1081.80 | 2025-07-31 09:15:00 | 1043.10 | EXIT_EMA400 | -38.70 |
| SELL | 2025-08-21 13:15:00 | 1023.00 | 2025-08-25 11:15:00 | 992.90 | TARGET | 30.10 |
| SELL | 2025-08-22 10:15:00 | 1021.20 | 2025-08-26 09:15:00 | 986.25 | TARGET | 34.95 |
| BUY | 2025-11-12 14:15:00 | 1064.90 | 2025-11-14 09:15:00 | 1032.50 | EXIT_EMA400 | -32.40 |
| SELL | 2026-02-04 09:15:00 | 774.85 | 2026-03-11 14:15:00 | 790.40 | EXIT_EMA400 | -15.55 |
| SELL | 2026-02-11 11:15:00 | 783.65 | 2026-03-11 14:15:00 | 790.40 | EXIT_EMA400 | -6.75 |
