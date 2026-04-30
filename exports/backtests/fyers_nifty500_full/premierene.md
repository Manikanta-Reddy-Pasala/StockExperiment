# Premier Energies Ltd. (PREMIERENE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-09-03 09:15:00 → 2026-04-30 15:15:00 (2867 bars)
- **Last close:** 1012.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 8 |
| ENTRY1 | 5 |
| ENTRY2 | 6 |
| EXIT | 5 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 2 / 9
- **Target hits / EMA400 exits:** 2 / 9
- **Total realized P&L (per unit):** -153.54
- **Avg P&L per closed trade:** -13.96

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 15:15:00 | 933.00 | 1168.70 | 1169.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 855.75 | 1165.59 | 1167.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 14:15:00 | 946.50 | 946.46 | 1002.26 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-21 09:15:00 | 923.70 | 946.23 | 1001.59 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 951.00 | 910.95 | 955.96 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-17 11:15:00 | 961.25 | 911.88 | 955.98 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 13:15:00 | 1122.00 | 978.58 | 978.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 15:15:00 | 1135.00 | 981.54 | 979.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 1026.40 | 1042.22 | 1020.22 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-30 11:15:00 | 1067.00 | 1030.32 | 1020.22 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 1051.90 | 1058.57 | 1041.92 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-23 11:15:00 | 1065.00 | 1057.73 | 1042.37 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 1048.00 | 1062.91 | 1046.82 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-29 10:15:00 | 1061.30 | 1062.65 | 1047.00 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 1055.20 | 1062.47 | 1047.07 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-30 09:15:00 | 1081.80 | 1062.33 | 1047.30 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-31 09:15:00 | 1042.80 | 1062.71 | 1048.01 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 1013.50 | 1037.51 | 1037.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 09:15:00 | 994.50 | 1036.03 | 1036.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 1029.80 | 1028.85 | 1032.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-21 13:15:00 | 1023.00 | 1029.59 | 1032.98 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 1023.00 | 1029.59 | 1032.98 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-22 10:15:00 | 1021.20 | 1029.36 | 1032.79 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 1012.30 | 1024.62 | 1029.96 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-28 12:15:00 | 1009.20 | 1024.36 | 1029.77 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-01 12:15:00 | 1031.20 | 1022.25 | 1028.31 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 12:15:00 | 1076.40 | 1030.51 | 1030.42 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 09:15:00 | 1018.80 | 1031.15 | 1031.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 1013.00 | 1028.95 | 1030.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 1027.90 | 1027.76 | 1029.34 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 13:15:00 | 1071.00 | 1030.61 | 1030.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 1096.40 | 1041.13 | 1036.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 10:15:00 | 1053.00 | 1055.78 | 1045.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-12 14:15:00 | 1064.10 | 1045.12 | 1041.40 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-14 09:15:00 | 1032.50 | 1045.62 | 1041.83 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 997.80 | 1038.08 | 1038.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 986.40 | 1037.57 | 1037.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 799.00 | 770.32 | 836.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-04 09:15:00 | 774.85 | 771.84 | 835.13 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 789.00 | 779.24 | 828.89 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-11 12:15:00 | 782.75 | 779.38 | 828.22 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 785.00 | 748.97 | 785.96 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-03-11 14:15:00 | 790.40 | 749.91 | 785.88 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 09:15:00 | 891.45 | 806.97 | 806.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 09:15:00 | 906.10 | 812.68 | 809.60 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-21 09:15:00 | 923.70 | 2025-04-17 11:15:00 | 961.25 | EXIT_EMA400 | -37.55 |
| BUY | 2025-06-30 11:15:00 | 1067.00 | 2025-07-31 09:15:00 | 1042.80 | EXIT_EMA400 | -24.20 |
| BUY | 2025-07-23 11:15:00 | 1065.00 | 2025-07-31 09:15:00 | 1042.80 | EXIT_EMA400 | -22.20 |
| BUY | 2025-07-29 10:15:00 | 1061.30 | 2025-07-31 09:15:00 | 1042.80 | EXIT_EMA400 | -18.50 |
| BUY | 2025-07-30 09:15:00 | 1081.80 | 2025-07-31 09:15:00 | 1042.80 | EXIT_EMA400 | -39.00 |
| SELL | 2025-08-21 13:15:00 | 1023.00 | 2025-08-25 10:15:00 | 993.07 | TARGET | 29.93 |
| SELL | 2025-08-22 10:15:00 | 1021.20 | 2025-08-26 09:15:00 | 986.42 | TARGET | 34.78 |
| SELL | 2025-08-28 12:15:00 | 1009.20 | 2025-09-01 12:15:00 | 1031.20 | EXIT_EMA400 | -22.00 |
| BUY | 2025-11-12 14:15:00 | 1064.10 | 2025-11-14 09:15:00 | 1032.50 | EXIT_EMA400 | -31.60 |
| SELL | 2026-02-04 09:15:00 | 774.85 | 2026-03-11 14:15:00 | 790.40 | EXIT_EMA400 | -15.55 |
| SELL | 2026-02-11 12:15:00 | 782.75 | 2026-03-11 14:15:00 | 790.40 | EXIT_EMA400 | -7.65 |
