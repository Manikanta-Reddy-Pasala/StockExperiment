# Anupam Rasayan India Ltd. (ANURAS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1332.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 6 |
| ENTRY1 | 6 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 1
- **Winners / losers:** 1 / 7
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -163.34
- **Avg P&L per closed trade:** -20.42

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 14:15:00 | 771.00 | 777.15 | 777.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 09:15:00 | 766.60 | 776.49 | 776.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 12:15:00 | 765.00 | 758.72 | 766.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-07 09:15:00 | 732.15 | 757.95 | 765.28 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 739.90 | 730.94 | 742.61 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-07 12:15:00 | 738.35 | 731.01 | 742.59 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 741.50 | 731.29 | 742.50 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-11 13:15:00 | 727.70 | 731.58 | 742.05 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 14:15:00 | 735.40 | 729.76 | 740.02 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-14 15:15:00 | 741.00 | 729.87 | 740.02 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 12:15:00 | 782.45 | 702.98 | 702.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 11:15:00 | 783.40 | 707.59 | 705.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-27 13:15:00 | 733.50 | 735.74 | 722.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-03-27 14:15:00 | 748.30 | 735.87 | 722.61 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-07 09:15:00 | 709.50 | 740.42 | 727.36 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 1075.90 | 1095.07 | 1095.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 15:15:00 | 1069.50 | 1093.21 | 1094.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 15:15:00 | 1094.40 | 1087.75 | 1091.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-10 09:15:00 | 1075.10 | 1087.62 | 1091.01 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 1078.40 | 1087.22 | 1090.72 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-11 09:15:00 | 1072.00 | 1087.04 | 1090.60 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1082.10 | 1082.86 | 1087.89 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-17 10:15:00 | 1095.10 | 1082.98 | 1087.93 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 10:15:00 | 1238.40 | 1092.31 | 1092.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 12:15:00 | 1246.40 | 1095.23 | 1093.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 15:15:00 | 1291.10 | 1299.43 | 1251.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-06 09:15:00 | 1314.00 | 1263.59 | 1246.82 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 1273.20 | 1289.23 | 1264.04 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-02-16 13:15:00 | 1257.70 | 1288.49 | 1264.17 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 12:15:00 | 1210.00 | 1257.40 | 1257.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 1204.40 | 1254.96 | 1256.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 12:15:00 | 1269.20 | 1252.95 | 1255.08 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-02 10:15:00 | 1240.30 | 1253.43 | 1255.27 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-02 13:15:00 | 1256.00 | 1253.28 | 1255.16 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 15:15:00 | 1283.50 | 1256.60 | 1256.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 1291.50 | 1258.45 | 1257.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 15:15:00 | 1291.00 | 1291.06 | 1276.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-30 09:15:00 | 1302.80 | 1291.18 | 1276.91 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-07 12:15:00 | 738.35 | 2024-11-13 09:15:00 | 725.64 | TARGET | 12.71 |
| SELL | 2024-10-07 09:15:00 | 732.15 | 2024-11-14 15:15:00 | 741.00 | EXIT_EMA400 | -8.85 |
| SELL | 2024-11-11 13:15:00 | 727.70 | 2024-11-14 15:15:00 | 741.00 | EXIT_EMA400 | -13.30 |
| BUY | 2025-03-27 14:15:00 | 748.30 | 2025-04-07 09:15:00 | 709.50 | EXIT_EMA400 | -38.80 |
| SELL | 2025-11-10 09:15:00 | 1075.10 | 2025-11-17 10:15:00 | 1095.10 | EXIT_EMA400 | -20.00 |
| SELL | 2025-11-11 09:15:00 | 1072.00 | 2025-11-17 10:15:00 | 1095.10 | EXIT_EMA400 | -23.10 |
| BUY | 2026-02-06 09:15:00 | 1314.00 | 2026-02-16 13:15:00 | 1257.70 | EXIT_EMA400 | -56.30 |
| SELL | 2026-04-02 10:15:00 | 1240.30 | 2026-04-02 13:15:00 | 1256.00 | EXIT_EMA400 | -15.70 |
