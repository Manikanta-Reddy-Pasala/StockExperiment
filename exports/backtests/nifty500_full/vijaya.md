# Vijaya Diagnostic Centre Ltd. (VIJAYA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1129.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 8 |
| ENTRY1 | 6 |
| ENTRY2 | 4 |
| EXIT | 6 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 2 / 8
- **Target hits / EMA400 exits:** 2 / 8
- **Total realized P&L (per unit):** -144.55
- **Avg P&L per closed trade:** -14.46

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 15:15:00 | 605.20 | 632.54 | 632.61 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-02 11:15:00 | 652.00 | 632.04 | 631.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 13:15:00 | 659.00 | 632.51 | 632.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-06 10:15:00 | 665.45 | 668.20 | 655.43 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-06 14:15:00 | 680.00 | 668.09 | 655.62 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 665.40 | 668.34 | 656.30 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-05-08 12:15:00 | 673.15 | 668.36 | 656.49 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 749.75 | 776.65 | 748.57 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-06-27 15:15:00 | 753.65 | 776.18 | 748.62 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 751.35 | 775.93 | 748.63 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-06-28 11:15:00 | 748.05 | 775.40 | 748.64 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-02-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 13:15:00 | 907.05 | 1060.70 | 1061.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-17 09:15:00 | 882.05 | 1056.19 | 1059.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 14:15:00 | 1050.60 | 1013.32 | 1034.13 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-27 09:15:00 | 993.00 | 1013.51 | 1034.02 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 993.00 | 1013.51 | 1034.02 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-27 10:15:00 | 955.85 | 1012.94 | 1033.63 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 1020.00 | 999.28 | 1024.38 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-04 11:15:00 | 1056.05 | 999.85 | 1024.54 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 11:15:00 | 1088.40 | 1024.39 | 1024.15 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 15:15:00 | 995.00 | 1024.19 | 1024.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 987.50 | 1019.61 | 1021.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 968.05 | 965.13 | 985.04 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-10 11:15:00 | 956.45 | 966.29 | 982.99 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-06-26 09:15:00 | 975.00 | 955.82 | 971.22 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 15:15:00 | 1009.05 | 980.78 | 980.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 11:15:00 | 1016.70 | 981.63 | 981.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 09:15:00 | 1042.70 | 1044.95 | 1022.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-13 09:15:00 | 1058.10 | 1045.99 | 1024.53 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1035.90 | 1048.09 | 1029.15 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-21 11:15:00 | 1029.00 | 1047.78 | 1029.18 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 13:15:00 | 1000.60 | 1033.34 | 1033.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 994.50 | 1031.99 | 1032.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 11:15:00 | 1013.55 | 1006.74 | 1016.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-30 13:15:00 | 994.85 | 1006.57 | 1016.05 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1005.00 | 1004.98 | 1014.77 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-03 13:15:00 | 1017.45 | 1005.16 | 1014.67 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 1063.20 | 1013.80 | 1013.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 12:15:00 | 1067.30 | 1017.93 | 1015.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 11:15:00 | 1022.60 | 1023.88 | 1019.19 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 10:15:00 | 990.00 | 1015.46 | 1015.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 11:15:00 | 976.50 | 1015.07 | 1015.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 990.00 | 983.18 | 995.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-25 12:15:00 | 982.15 | 996.08 | 999.06 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 996.05 | 995.85 | 998.88 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-26 13:15:00 | 987.10 | 995.63 | 998.71 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-27 14:15:00 | 998.90 | 994.63 | 998.08 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 13:15:00 | 1024.45 | 968.51 | 968.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 14:15:00 | 1030.00 | 969.12 | 968.68 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-05-08 12:15:00 | 673.15 | 2024-05-09 09:15:00 | 723.12 | TARGET | 49.97 |
| BUY | 2024-05-06 14:15:00 | 680.00 | 2024-05-10 09:15:00 | 753.13 | TARGET | 73.13 |
| BUY | 2024-06-27 15:15:00 | 753.65 | 2024-06-28 11:15:00 | 748.05 | EXIT_EMA400 | -5.60 |
| SELL | 2025-02-27 09:15:00 | 993.00 | 2025-03-04 11:15:00 | 1056.05 | EXIT_EMA400 | -63.05 |
| SELL | 2025-02-27 10:15:00 | 955.85 | 2025-03-04 11:15:00 | 1056.05 | EXIT_EMA400 | -100.20 |
| SELL | 2025-06-10 11:15:00 | 956.45 | 2025-06-26 09:15:00 | 975.00 | EXIT_EMA400 | -18.55 |
| BUY | 2025-08-13 09:15:00 | 1058.10 | 2025-08-21 11:15:00 | 1029.00 | EXIT_EMA400 | -29.10 |
| SELL | 2025-10-30 13:15:00 | 994.85 | 2025-11-03 13:15:00 | 1017.45 | EXIT_EMA400 | -22.60 |
| SELL | 2026-02-25 12:15:00 | 982.15 | 2026-02-27 14:15:00 | 998.90 | EXIT_EMA400 | -16.75 |
| SELL | 2026-02-26 13:15:00 | 987.10 | 2026-02-27 14:15:00 | 998.90 | EXIT_EMA400 | -11.80 |
