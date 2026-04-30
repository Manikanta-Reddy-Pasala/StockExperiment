# Kirloskar Oil Eng Ltd. (KIRLOSENG.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1708.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 6 |
| ENTRY1 | 2 |
| ENTRY2 | 4 |
| EXIT | 2 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / EMA400 exits:** 4 / 2
- **Total realized P&L (per unit):** 198.55
- **Avg P&L per closed trade:** 33.09

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 10:15:00 | 1144.25 | 1251.30 | 1251.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-07 11:15:00 | 1138.65 | 1250.18 | 1250.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 10:15:00 | 1257.05 | 1239.85 | 1245.27 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 13:15:00 | 1337.60 | 1250.31 | 1249.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 14:15:00 | 1350.00 | 1251.31 | 1250.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-09 09:15:00 | 1289.90 | 1297.56 | 1278.83 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 15:15:00 | 1203.10 | 1271.16 | 1271.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 09:15:00 | 1182.30 | 1264.64 | 1267.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 14:15:00 | 1226.95 | 1217.66 | 1239.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-16 12:15:00 | 1210.00 | 1217.75 | 1238.83 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 1209.30 | 1216.68 | 1237.46 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-10-17 14:15:00 | 1205.80 | 1216.57 | 1237.30 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 1175.75 | 1155.97 | 1192.77 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-06 13:15:00 | 1199.85 | 1156.76 | 1192.80 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 10:15:00 | 876.70 | 763.51 | 763.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 878.25 | 780.33 | 772.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 848.00 | 849.46 | 818.52 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-17 09:15:00 | 884.75 | 850.59 | 820.15 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 833.85 | 851.08 | 833.30 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-09 09:15:00 | 839.10 | 850.46 | 833.34 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 833.85 | 849.76 | 833.41 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-10 09:15:00 | 845.60 | 849.56 | 833.47 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 869.70 | 893.96 | 869.68 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-04 14:15:00 | 886.25 | 892.89 | 869.86 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 873.25 | 892.45 | 869.98 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-05 12:15:00 | 867.45 | 891.99 | 869.97 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-16 12:15:00 | 1210.00 | 2024-10-22 09:15:00 | 1123.52 | TARGET | 86.48 |
| SELL | 2024-10-17 14:15:00 | 1205.80 | 2024-10-22 10:15:00 | 1111.30 | TARGET | 94.50 |
| BUY | 2025-07-09 09:15:00 | 839.10 | 2025-07-10 10:15:00 | 856.38 | TARGET | 17.28 |
| BUY | 2025-07-10 09:15:00 | 845.60 | 2025-07-10 13:15:00 | 881.99 | TARGET | 36.39 |
| BUY | 2025-06-17 09:15:00 | 884.75 | 2025-08-05 12:15:00 | 867.45 | EXIT_EMA400 | -17.30 |
| BUY | 2025-08-04 14:15:00 | 886.25 | 2025-08-05 12:15:00 | 867.45 | EXIT_EMA400 | -18.80 |
