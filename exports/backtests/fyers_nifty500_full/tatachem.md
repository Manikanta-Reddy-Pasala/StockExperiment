# Tata Chemicals Ltd. (TATACHEM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 806.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 5 / 3
- **Target hits / EMA400 exits:** 5 / 3
- **Total realized P&L (per unit):** 196.01
- **Avg P&L per closed trade:** 24.50

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 15:15:00 | 1033.05 | 1074.30 | 1074.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 1029.60 | 1068.96 | 1071.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 09:15:00 | 1066.60 | 1061.50 | 1067.14 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 14:15:00 | 1109.00 | 1071.06 | 1070.90 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 14:15:00 | 1050.70 | 1070.93 | 1070.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 13:15:00 | 1035.40 | 1069.36 | 1070.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-24 13:15:00 | 1053.20 | 1052.99 | 1060.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-24 15:15:00 | 1048.95 | 1052.92 | 1060.42 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 1059.85 | 1052.99 | 1060.42 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-09-25 10:15:00 | 1060.70 | 1053.07 | 1060.42 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 10:15:00 | 1147.00 | 1066.24 | 1065.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 1163.05 | 1075.74 | 1071.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 11:15:00 | 1088.60 | 1091.32 | 1080.65 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-21 09:15:00 | 1196.90 | 1090.54 | 1080.85 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-25 09:15:00 | 1086.75 | 1103.38 | 1089.14 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 1037.90 | 1094.41 | 1094.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 1025.75 | 1093.73 | 1094.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 10:15:00 | 1092.30 | 1089.07 | 1091.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-24 14:15:00 | 1067.50 | 1088.53 | 1091.36 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-24 10:15:00 | 871.50 | 842.54 | 868.54 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 14:15:00 | 905.20 | 865.01 | 864.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 12:15:00 | 908.95 | 867.01 | 865.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 904.10 | 905.85 | 890.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 10:15:00 | 917.75 | 906.22 | 891.16 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 906.65 | 920.39 | 905.79 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-11 15:15:00 | 910.40 | 920.29 | 905.82 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 919.75 | 920.29 | 905.89 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-14 12:15:00 | 922.50 | 920.28 | 906.10 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 937.45 | 947.93 | 933.06 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-18 09:15:00 | 949.00 | 947.67 | 933.15 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 935.40 | 946.23 | 935.12 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-26 14:15:00 | 933.70 | 945.95 | 935.25 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 11:15:00 | 907.65 | 940.70 | 940.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 12:15:00 | 907.05 | 940.37 | 940.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 13:15:00 | 776.15 | 774.89 | 808.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 10:15:00 | 768.40 | 775.01 | 808.04 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-10 15:15:00 | 705.00 | 656.91 | 687.02 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 816.00 | 703.45 | 703.33 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-24 15:15:00 | 1048.95 | 2024-09-25 10:15:00 | 1060.70 | EXIT_EMA400 | -11.75 |
| BUY | 2024-10-21 09:15:00 | 1196.90 | 2024-10-25 09:15:00 | 1086.75 | EXIT_EMA400 | -110.15 |
| SELL | 2024-12-24 14:15:00 | 1067.50 | 2025-01-06 10:15:00 | 995.93 | TARGET | 71.57 |
| BUY | 2025-07-11 15:15:00 | 910.40 | 2025-07-14 13:15:00 | 924.15 | TARGET | 13.75 |
| BUY | 2025-07-14 12:15:00 | 922.50 | 2025-07-23 10:15:00 | 971.71 | TARGET | 49.21 |
| BUY | 2025-06-24 10:15:00 | 917.75 | 2025-07-29 10:15:00 | 997.52 | TARGET | 79.77 |
| BUY | 2025-08-18 09:15:00 | 949.00 | 2025-08-26 14:15:00 | 933.70 | EXIT_EMA400 | -15.30 |
| SELL | 2026-01-08 10:15:00 | 768.40 | 2026-03-18 15:15:00 | 649.48 | TARGET | 118.92 |
