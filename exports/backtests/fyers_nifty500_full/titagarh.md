# Titagarh Rail Systems Ltd. (TITAGARH.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 771.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 6 |
| ENTRY1 | 6 |
| ENTRY2 | 3 |
| EXIT | 6 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 3 / 6
- **Target hits / EMA400 exits:** 2 / 7
- **Total realized P&L (per unit):** 13.16
- **Avg P&L per closed trade:** 1.46

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 14:15:00 | 1412.40 | 1466.66 | 1466.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 09:15:00 | 1402.95 | 1465.52 | 1466.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 12:15:00 | 1207.80 | 1193.66 | 1265.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 14:15:00 | 1151.95 | 1195.44 | 1251.67 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-28 09:15:00 | 1234.70 | 1164.18 | 1214.06 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 14:15:00 | 1318.10 | 1234.22 | 1234.09 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 12:15:00 | 1183.00 | 1233.92 | 1234.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 14:15:00 | 1176.60 | 1232.84 | 1233.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 09:15:00 | 1095.65 | 1054.75 | 1113.79 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-01 12:15:00 | 954.00 | 1053.84 | 1112.46 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 795.90 | 757.59 | 797.13 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-05-14 14:15:00 | 805.35 | 758.06 | 797.17 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 14:15:00 | 938.80 | 825.84 | 825.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 942.75 | 863.95 | 847.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 881.60 | 884.78 | 861.52 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 09:15:00 | 906.40 | 880.97 | 863.89 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 907.95 | 925.39 | 904.37 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-24 15:15:00 | 904.00 | 924.98 | 904.37 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 09:15:00 | 834.75 | 890.39 | 890.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 11:15:00 | 826.55 | 889.22 | 889.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 13:15:00 | 858.90 | 857.61 | 871.36 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-22 10:15:00 | 845.50 | 857.41 | 870.99 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 869.20 | 857.64 | 870.57 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-25 12:15:00 | 861.85 | 857.68 | 870.53 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 868.20 | 858.11 | 870.30 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-26 13:15:00 | 865.70 | 858.19 | 870.28 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 867.50 | 858.39 | 870.26 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-28 09:15:00 | 852.60 | 858.33 | 870.17 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-09 10:15:00 | 874.00 | 852.54 | 863.88 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 13:15:00 | 934.65 | 873.48 | 873.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 14:15:00 | 937.70 | 874.12 | 873.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 11:15:00 | 888.85 | 895.49 | 886.07 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-29 10:15:00 | 896.15 | 894.96 | 886.08 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-29 11:15:00 | 885.75 | 894.87 | 886.08 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 11:15:00 | 846.65 | 885.87 | 885.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 12:15:00 | 846.00 | 885.48 | 885.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 886.95 | 882.66 | 884.31 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-13 13:15:00 | 864.00 | 882.04 | 883.91 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 863.70 | 881.50 | 883.61 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-17 09:15:00 | 893.80 | 880.75 | 883.16 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-08 14:15:00 | 1151.95 | 2024-11-28 09:15:00 | 1234.70 | EXIT_EMA400 | -82.75 |
| SELL | 2025-02-01 12:15:00 | 954.00 | 2025-05-14 14:15:00 | 805.35 | EXIT_EMA400 | 148.65 |
| BUY | 2025-06-24 09:15:00 | 906.40 | 2025-07-24 15:15:00 | 904.00 | EXIT_EMA400 | -2.40 |
| SELL | 2025-08-26 13:15:00 | 865.70 | 2025-08-28 09:15:00 | 851.97 | TARGET | 13.73 |
| SELL | 2025-08-25 12:15:00 | 861.85 | 2025-08-29 12:15:00 | 835.82 | TARGET | 26.03 |
| SELL | 2025-08-22 10:15:00 | 845.50 | 2025-09-09 10:15:00 | 874.00 | EXIT_EMA400 | -28.50 |
| SELL | 2025-08-28 09:15:00 | 852.60 | 2025-09-09 10:15:00 | 874.00 | EXIT_EMA400 | -21.40 |
| BUY | 2025-09-29 10:15:00 | 896.15 | 2025-09-29 11:15:00 | 885.75 | EXIT_EMA400 | -10.40 |
| SELL | 2025-11-13 13:15:00 | 864.00 | 2025-11-17 09:15:00 | 893.80 | EXIT_EMA400 | -29.80 |
