# Vijaya Diagnostic Centre Ltd. (VIJAYA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1131.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 5 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -140.96
- **Avg P&L per closed trade:** -20.14

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 11:15:00 | 839.05 | 785.90 | 785.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-13 09:15:00 | 857.65 | 794.61 | 790.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 12:15:00 | 897.15 | 897.75 | 869.46 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-30 12:15:00 | 908.80 | 897.64 | 870.37 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 14:15:00 | 937.90 | 954.31 | 919.79 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-25 09:15:00 | 914.55 | 953.67 | 919.82 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-02-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 13:15:00 | 907.05 | 1061.18 | 1061.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-17 09:15:00 | 882.05 | 1056.63 | 1059.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 14:15:00 | 1050.60 | 1013.54 | 1034.50 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-27 09:15:00 | 992.65 | 1013.73 | 1034.38 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 992.65 | 1013.73 | 1034.38 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-27 10:15:00 | 955.85 | 1013.15 | 1033.99 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 1020.00 | 999.54 | 1024.75 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-04 11:15:00 | 1056.05 | 1000.11 | 1024.91 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 11:15:00 | 1089.00 | 1024.40 | 1024.24 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 14:15:00 | 992.60 | 1024.52 | 1024.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 987.50 | 1019.61 | 1021.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 968.05 | 965.03 | 985.02 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-10 11:15:00 | 956.45 | 966.25 | 982.99 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-06-26 09:15:00 | 975.00 | 955.83 | 971.24 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 15:15:00 | 1009.95 | 980.92 | 980.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 11:15:00 | 1016.70 | 981.76 | 981.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 09:15:00 | 1042.70 | 1044.87 | 1022.44 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-13 09:15:00 | 1058.10 | 1045.92 | 1024.49 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1035.90 | 1047.99 | 1029.09 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-21 11:15:00 | 1029.00 | 1047.67 | 1029.12 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 14:15:00 | 1000.00 | 1033.15 | 1033.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 994.50 | 1032.14 | 1032.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 11:15:00 | 1013.55 | 1006.76 | 1016.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-30 13:15:00 | 994.85 | 1006.60 | 1016.07 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1005.00 | 1005.02 | 1014.80 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-03 13:15:00 | 1017.45 | 1005.20 | 1014.70 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 1063.20 | 1013.77 | 1013.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 12:15:00 | 1067.20 | 1017.90 | 1015.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 11:15:00 | 1022.60 | 1023.80 | 1019.13 | EMA200 retest candle locked |

### Cycle 8 — SELL (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 10:15:00 | 990.00 | 1015.45 | 1015.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 11:15:00 | 976.50 | 1015.06 | 1015.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 990.00 | 981.67 | 994.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-27 09:15:00 | 976.50 | 994.58 | 997.64 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-27 14:15:00 | 998.90 | 994.00 | 997.28 | Close above EMA400 |

### Cycle 9 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 1035.20 | 968.00 | 967.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 13:15:00 | 1050.45 | 974.64 | 971.27 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-09-30 12:15:00 | 908.80 | 2024-10-15 10:15:00 | 1024.09 | TARGET | 115.29 |
| SELL | 2025-02-27 09:15:00 | 992.65 | 2025-03-04 11:15:00 | 1056.05 | EXIT_EMA400 | -63.40 |
| SELL | 2025-02-27 10:15:00 | 955.85 | 2025-03-04 11:15:00 | 1056.05 | EXIT_EMA400 | -100.20 |
| SELL | 2025-06-10 11:15:00 | 956.45 | 2025-06-26 09:15:00 | 975.00 | EXIT_EMA400 | -18.55 |
| BUY | 2025-08-13 09:15:00 | 1058.10 | 2025-08-21 11:15:00 | 1029.00 | EXIT_EMA400 | -29.10 |
| SELL | 2025-10-30 13:15:00 | 994.85 | 2025-11-03 13:15:00 | 1017.45 | EXIT_EMA400 | -22.60 |
| SELL | 2026-02-27 09:15:00 | 976.50 | 2026-02-27 14:15:00 | 998.90 | EXIT_EMA400 | -22.40 |
