# Max Healthcare Institute Ltd. (MAXHEALTH.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 993.05
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 4 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 171.08
- **Avg P&L per closed trade:** 24.44

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 14:15:00 | 594.15 | 568.92 | 568.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 15:15:00 | 596.00 | 569.19 | 568.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 14:15:00 | 569.60 | 580.91 | 575.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-10-03 09:15:00 | 593.05 | 572.42 | 572.20 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 593.05 | 572.42 | 572.20 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-10-03 11:15:00 | 599.00 | 572.91 | 572.45 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2023-10-04 09:15:00 | 568.45 | 573.70 | 572.86 | Close below EMA400 |

### Cycle 2 — SELL (started 2023-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 14:15:00 | 569.40 | 572.13 | 572.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-10 10:15:00 | 566.15 | 571.98 | 572.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-11 09:15:00 | 576.30 | 571.56 | 571.84 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2023-10-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 10:15:00 | 588.25 | 572.20 | 572.15 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-27 10:15:00 | 562.50 | 572.67 | 572.67 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-11-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-01 14:15:00 | 576.40 | 572.65 | 572.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 11:15:00 | 586.45 | 573.01 | 572.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-21 09:15:00 | 645.40 | 648.89 | 623.55 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-21 10:15:00 | 652.40 | 648.92 | 623.69 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-01 09:15:00 | 733.30 | 814.02 | 766.31 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 13:15:00 | 1006.90 | 1072.06 | 1072.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 11:15:00 | 985.85 | 1048.51 | 1058.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 15:15:00 | 1017.50 | 1017.04 | 1036.44 | EMA200 retest candle locked |

### Cycle 7 — BUY (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 14:15:00 | 1173.45 | 1053.59 | 1053.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 1175.50 | 1111.88 | 1093.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 09:15:00 | 1133.70 | 1140.37 | 1116.54 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-02 12:15:00 | 1156.80 | 1139.96 | 1118.26 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1221.50 | 1236.80 | 1202.93 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-21 12:15:00 | 1224.90 | 1236.35 | 1203.21 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-14 09:15:00 | 1227.40 | 1251.90 | 1228.07 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 10:15:00 | 1167.90 | 1215.92 | 1216.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 12:15:00 | 1164.20 | 1214.96 | 1215.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 1154.70 | 1152.81 | 1174.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-10 09:15:00 | 1144.10 | 1152.91 | 1173.61 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1166.10 | 1153.06 | 1171.53 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-15 11:15:00 | 1159.50 | 1153.13 | 1171.47 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-16 13:15:00 | 1172.10 | 1153.99 | 1171.10 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-10-03 09:15:00 | 593.05 | 2023-10-04 09:15:00 | 568.45 | EXIT_EMA400 | -24.60 |
| BUY | 2023-10-03 11:15:00 | 599.00 | 2023-10-04 09:15:00 | 568.45 | EXIT_EMA400 | -30.55 |
| BUY | 2023-12-21 10:15:00 | 652.40 | 2024-01-04 14:15:00 | 738.52 | TARGET | 86.12 |
| BUY | 2025-06-02 12:15:00 | 1156.80 | 2025-06-26 13:15:00 | 1272.43 | TARGET | 115.63 |
| BUY | 2025-07-21 12:15:00 | 1224.90 | 2025-08-13 09:15:00 | 1289.98 | TARGET | 65.08 |
| SELL | 2025-10-10 09:15:00 | 1144.10 | 2025-10-16 13:15:00 | 1172.10 | EXIT_EMA400 | -28.00 |
| SELL | 2025-10-15 11:15:00 | 1159.50 | 2025-10-16 13:15:00 | 1172.10 | EXIT_EMA400 | -12.60 |
