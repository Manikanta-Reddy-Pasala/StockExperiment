# Deepak Fertilisers & Petrochemicals Corp. Ltd. (DEEPAKFERT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1242.75
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / EMA400 exits:** 3 / 2
- **Total realized P&L (per unit):** 409.70
- **Avg P&L per closed trade:** 81.94

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 12:15:00 | 1155.80 | 1197.05 | 1197.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 1145.00 | 1195.74 | 1196.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1200.65 | 1192.85 | 1194.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-27 09:15:00 | 1139.10 | 1190.79 | 1193.74 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-29 10:15:00 | 1189.55 | 1182.22 | 1189.07 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 12:15:00 | 1245.00 | 1122.37 | 1121.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 13:15:00 | 1252.40 | 1123.66 | 1122.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 1221.20 | 1223.41 | 1184.64 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-07 10:15:00 | 1254.00 | 1223.71 | 1184.99 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-28 09:15:00 | 1510.20 | 1583.83 | 1516.50 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 12:15:00 | 1394.40 | 1501.64 | 1501.80 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 11:15:00 | 1564.00 | 1485.14 | 1485.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 1575.70 | 1487.56 | 1486.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 10:15:00 | 1496.80 | 1500.23 | 1493.43 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-10 13:15:00 | 1506.00 | 1500.26 | 1493.54 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 1499.30 | 1500.25 | 1493.57 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-10-10 15:15:00 | 1493.00 | 1500.18 | 1493.57 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 11:15:00 | 1430.70 | 1487.87 | 1487.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 10:15:00 | 1419.60 | 1481.72 | 1484.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 1521.40 | 1471.77 | 1478.88 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-06 09:15:00 | 1438.20 | 1480.65 | 1482.51 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1447.40 | 1447.59 | 1462.31 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-21 09:15:00 | 1417.30 | 1447.36 | 1461.69 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1017.40 | 974.73 | 1033.95 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-08 14:15:00 | 1050.70 | 977.42 | 1033.85 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 13:15:00 | 1267.75 | 1073.09 | 1072.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 14:15:00 | 1270.05 | 1075.05 | 1073.23 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-27 09:15:00 | 1139.10 | 2025-01-29 10:15:00 | 1189.55 | EXIT_EMA400 | -50.45 |
| BUY | 2025-05-07 10:15:00 | 1254.00 | 2025-05-29 09:15:00 | 1461.04 | TARGET | 207.04 |
| BUY | 2025-10-10 13:15:00 | 1506.00 | 2025-10-10 15:15:00 | 1493.00 | EXIT_EMA400 | -13.00 |
| SELL | 2025-11-06 09:15:00 | 1438.20 | 2025-12-03 09:15:00 | 1305.26 | TARGET | 132.94 |
| SELL | 2025-11-21 09:15:00 | 1417.30 | 2025-12-08 09:15:00 | 1284.14 | TARGET | 133.16 |
