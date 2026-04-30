# UTI Asset Management Company Ltd. (UTIAMC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 950.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / EMA400 exits:** 4 / 3
- **Total realized P&L (per unit):** 81.91
- **Avg P&L per closed trade:** 11.70

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 15:15:00 | 751.45 | 774.99 | 774.99 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 12:15:00 | 785.00 | 774.98 | 774.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-13 09:15:00 | 787.90 | 775.97 | 775.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-13 11:15:00 | 774.05 | 776.03 | 775.51 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-11-16 10:15:00 | 785.90 | 776.10 | 775.58 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 15:15:00 | 775.95 | 776.26 | 775.67 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-11-17 09:15:00 | 783.90 | 776.34 | 775.71 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 14:15:00 | 779.00 | 779.03 | 777.23 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-11-23 10:15:00 | 782.75 | 779.08 | 777.28 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-01-18 09:15:00 | 836.50 | 860.49 | 837.85 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-21 09:15:00 | 822.50 | 872.37 | 872.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 14:15:00 | 812.15 | 862.11 | 866.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-03 10:15:00 | 857.40 | 856.50 | 863.38 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 12:15:00 | 933.80 | 869.64 | 869.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 12:15:00 | 943.90 | 885.66 | 878.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 10:15:00 | 914.60 | 916.67 | 898.73 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-08 09:15:00 | 920.85 | 916.33 | 899.09 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-05-09 14:15:00 | 898.30 | 916.40 | 900.14 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 12:15:00 | 1221.25 | 1273.29 | 1273.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 14:15:00 | 1215.70 | 1272.19 | 1272.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 14:15:00 | 987.40 | 985.33 | 1055.05 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-07 09:15:00 | 976.00 | 1041.73 | 1051.66 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-12 15:15:00 | 1048.00 | 1032.84 | 1045.61 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 11:15:00 | 1172.00 | 1056.32 | 1055.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 12:15:00 | 1187.60 | 1057.63 | 1056.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 11:15:00 | 1341.90 | 1350.25 | 1275.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-25 14:15:00 | 1355.80 | 1350.14 | 1276.85 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-28 11:15:00 | 1312.40 | 1348.86 | 1313.10 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 09:15:00 | 1243.50 | 1325.66 | 1325.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 10:15:00 | 1236.60 | 1319.46 | 1322.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 14:15:00 | 1141.00 | 1139.33 | 1176.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-06 09:15:00 | 1127.20 | 1139.20 | 1176.51 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-11 09:15:00 | 1106.40 | 1062.50 | 1103.19 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-11-23 10:15:00 | 782.75 | 2023-11-28 09:15:00 | 799.15 | TARGET | 16.40 |
| BUY | 2023-11-16 10:15:00 | 785.90 | 2023-11-30 09:15:00 | 816.87 | TARGET | 30.97 |
| BUY | 2023-11-17 09:15:00 | 783.90 | 2023-11-30 09:15:00 | 808.46 | TARGET | 24.56 |
| BUY | 2024-05-08 09:15:00 | 920.85 | 2024-05-09 14:15:00 | 898.30 | EXIT_EMA400 | -22.55 |
| SELL | 2025-05-07 09:15:00 | 976.00 | 2025-05-12 15:15:00 | 1048.00 | EXIT_EMA400 | -72.00 |
| BUY | 2025-07-25 14:15:00 | 1355.80 | 2025-08-28 11:15:00 | 1312.40 | EXIT_EMA400 | -43.40 |
| SELL | 2026-01-06 09:15:00 | 1127.20 | 2026-01-23 13:15:00 | 979.26 | TARGET | 147.94 |
