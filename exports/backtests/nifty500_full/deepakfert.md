# Deepak Fertilisers & Petrochemicals Corp. Ltd. (DEEPAKFERT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1265.25
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT3 | 3 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / EMA400 exits:** 4 / 3
- **Total realized P&L (per unit):** 451.04
- **Avg P&L per closed trade:** 64.43

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 11:15:00 | 604.90 | 563.55 | 563.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 09:15:00 | 613.55 | 570.68 | 567.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 10:15:00 | 626.00 | 626.63 | 607.62 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-10-10 09:15:00 | 631.45 | 626.53 | 608.13 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2023-10-23 12:15:00 | 620.00 | 642.71 | 622.43 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 10:15:00 | 577.95 | 645.01 | 645.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-02 11:15:00 | 573.40 | 644.30 | 644.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 10:15:00 | 513.00 | 507.95 | 540.65 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2024-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-29 12:15:00 | 612.30 | 552.05 | 551.80 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 10:15:00 | 546.00 | 554.08 | 554.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 11:15:00 | 543.50 | 553.97 | 554.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 14:15:00 | 574.00 | 553.99 | 554.04 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2024-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 15:15:00 | 567.70 | 554.13 | 554.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 12:15:00 | 580.00 | 555.50 | 554.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 550.45 | 556.59 | 555.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-06 09:15:00 | 573.35 | 554.81 | 554.55 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-21 11:15:00 | 988.00 | 1041.48 | 993.73 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 13:15:00 | 1161.05 | 1196.57 | 1196.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 1143.75 | 1195.61 | 1196.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1200.65 | 1192.73 | 1194.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-27 09:15:00 | 1139.10 | 1190.65 | 1193.54 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-29 10:15:00 | 1189.55 | 1182.12 | 1188.90 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 12:15:00 | 1245.00 | 1122.48 | 1122.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 13:15:00 | 1252.40 | 1123.77 | 1122.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 1221.20 | 1223.38 | 1184.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-07 10:15:00 | 1252.80 | 1223.67 | 1185.12 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-28 09:15:00 | 1510.20 | 1583.79 | 1516.52 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 12:15:00 | 1394.40 | 1501.76 | 1501.91 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 11:15:00 | 1564.00 | 1485.16 | 1485.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 1575.90 | 1487.58 | 1486.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 10:15:00 | 1496.80 | 1500.28 | 1493.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-10 13:15:00 | 1506.00 | 1500.31 | 1493.60 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 1499.30 | 1500.30 | 1493.62 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-10-10 15:15:00 | 1493.00 | 1500.22 | 1493.62 | Close below EMA400 |

### Cycle 10 — SELL (started 2025-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 12:15:00 | 1434.00 | 1487.42 | 1487.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 10:15:00 | 1419.60 | 1481.82 | 1484.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 1521.40 | 1471.83 | 1478.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-06 09:15:00 | 1438.20 | 1480.75 | 1482.60 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1447.40 | 1447.64 | 1462.37 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-21 09:15:00 | 1417.30 | 1447.40 | 1461.74 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1017.40 | 975.36 | 1035.82 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-08 14:15:00 | 1050.70 | 978.03 | 1035.68 | Close above EMA400 |

### Cycle 11 — BUY (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 10:15:00 | 1237.60 | 1073.95 | 1073.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 11:15:00 | 1272.60 | 1075.93 | 1074.87 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-10-10 09:15:00 | 631.45 | 2023-10-23 12:15:00 | 620.00 | EXIT_EMA400 | -11.45 |
| BUY | 2024-06-06 09:15:00 | 573.35 | 2024-06-19 09:15:00 | 629.74 | TARGET | 56.39 |
| SELL | 2025-01-27 09:15:00 | 1139.10 | 2025-01-29 10:15:00 | 1189.55 | EXIT_EMA400 | -50.45 |
| BUY | 2025-05-07 10:15:00 | 1252.80 | 2025-05-29 09:15:00 | 1455.84 | TARGET | 203.04 |
| BUY | 2025-10-10 13:15:00 | 1506.00 | 2025-10-10 15:15:00 | 1493.00 | EXIT_EMA400 | -13.00 |
| SELL | 2025-11-06 09:15:00 | 1438.20 | 2025-12-03 09:15:00 | 1305.01 | TARGET | 133.19 |
| SELL | 2025-11-21 09:15:00 | 1417.30 | 2025-12-08 09:15:00 | 1283.97 | TARGET | 133.33 |
