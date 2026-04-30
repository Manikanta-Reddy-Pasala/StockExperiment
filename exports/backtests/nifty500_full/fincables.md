# Finolex Cables Ltd. (FINCABLES.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 990.55
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 144.38
- **Avg P&L per closed trade:** 20.63

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 11:15:00 | 868.70 | 1022.79 | 1023.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-26 09:15:00 | 846.25 | 1015.21 | 1019.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 09:15:00 | 938.75 | 936.95 | 963.67 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 10:15:00 | 1086.50 | 979.87 | 979.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-12 11:15:00 | 1099.95 | 981.07 | 980.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 14:15:00 | 1040.20 | 1042.09 | 1020.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-03 10:15:00 | 1045.10 | 1042.09 | 1020.43 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-01-09 13:15:00 | 1022.25 | 1040.52 | 1022.73 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-23 10:15:00 | 981.95 | 1039.04 | 1039.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-23 11:15:00 | 980.70 | 1038.46 | 1039.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 10:15:00 | 932.70 | 931.68 | 970.00 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 09:15:00 | 1028.35 | 985.08 | 984.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-30 10:15:00 | 1054.00 | 985.76 | 985.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-13 09:15:00 | 1000.35 | 1009.50 | 999.09 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-14 09:15:00 | 1028.45 | 1009.83 | 999.61 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-08-05 10:15:00 | 1464.45 | 1548.20 | 1465.26 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 13:15:00 | 1414.00 | 1451.63 | 1451.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 11:15:00 | 1407.00 | 1449.91 | 1450.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 10:15:00 | 1443.00 | 1435.91 | 1443.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-30 11:15:00 | 1427.80 | 1444.98 | 1447.03 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-05 11:15:00 | 1267.00 | 1193.25 | 1254.48 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 12:15:00 | 984.00 | 923.30 | 923.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 14:15:00 | 987.35 | 924.52 | 923.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 947.00 | 955.18 | 942.84 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-30 09:15:00 | 973.75 | 950.02 | 943.53 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 954.00 | 958.90 | 950.92 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-14 12:15:00 | 950.05 | 958.68 | 950.93 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 12:15:00 | 903.10 | 945.78 | 945.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 15:15:00 | 897.20 | 944.42 | 945.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 11:15:00 | 863.60 | 862.28 | 889.42 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-03 12:15:00 | 855.00 | 862.15 | 888.29 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 778.40 | 757.67 | 780.78 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-17 10:15:00 | 771.80 | 758.41 | 780.69 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-12-19 09:15:00 | 784.15 | 760.13 | 780.18 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 808.50 | 768.78 | 768.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 817.35 | 769.69 | 769.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 09:15:00 | 852.30 | 853.23 | 820.98 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-18 09:15:00 | 865.60 | 852.73 | 822.89 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-23 11:15:00 | 824.80 | 854.83 | 827.26 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-03 10:15:00 | 1045.10 | 2024-01-09 13:15:00 | 1022.25 | EXIT_EMA400 | -22.85 |
| BUY | 2024-05-14 09:15:00 | 1028.45 | 2024-05-22 13:15:00 | 1114.97 | TARGET | 86.52 |
| SELL | 2024-09-30 11:15:00 | 1427.80 | 2024-10-03 13:15:00 | 1370.11 | TARGET | 57.69 |
| BUY | 2025-06-30 09:15:00 | 973.75 | 2025-07-14 12:15:00 | 950.05 | EXIT_EMA400 | -23.70 |
| SELL | 2025-09-03 12:15:00 | 855.00 | 2025-11-07 09:15:00 | 755.13 | TARGET | 99.87 |
| SELL | 2025-12-17 10:15:00 | 771.80 | 2025-12-19 09:15:00 | 784.15 | EXIT_EMA400 | -12.35 |
| BUY | 2026-03-18 09:15:00 | 865.60 | 2026-03-23 11:15:00 | 824.80 | EXIT_EMA400 | -40.80 |
