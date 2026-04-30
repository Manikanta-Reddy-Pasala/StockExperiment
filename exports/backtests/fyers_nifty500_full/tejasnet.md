# Tejas Networks Ltd. (TEJASNET.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 413.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 7 |
| ENTRY1 | 4 |
| ENTRY2 | 4 |
| EXIT | 4 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 4 / 4
- **Target hits / EMA400 exits:** 3 / 5
- **Total realized P&L (per unit):** 370.78
- **Avg P&L per closed trade:** 46.35

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 12:15:00 | 1214.85 | 1276.24 | 1276.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-16 14:15:00 | 1206.05 | 1274.89 | 1275.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 10:15:00 | 1276.30 | 1268.18 | 1272.23 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-26 14:15:00 | 1262.00 | 1273.39 | 1274.54 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 1262.00 | 1273.39 | 1274.54 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-08-27 09:15:00 | 1251.10 | 1273.09 | 1274.38 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 1274.00 | 1272.73 | 1274.18 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-08-27 13:15:00 | 1310.05 | 1273.11 | 1274.36 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 11:15:00 | 1325.20 | 1275.98 | 1275.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 10:15:00 | 1334.70 | 1280.17 | 1277.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-09 09:15:00 | 1285.85 | 1295.85 | 1287.04 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 11:15:00 | 1217.80 | 1281.11 | 1281.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 12:15:00 | 1199.25 | 1280.29 | 1280.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 09:15:00 | 1348.10 | 1210.51 | 1235.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-21 11:15:00 | 1276.05 | 1212.47 | 1235.98 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 1276.05 | 1212.47 | 1235.98 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-10-21 12:15:00 | 1299.05 | 1213.33 | 1236.29 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 14:15:00 | 1308.30 | 1252.35 | 1252.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 1361.80 | 1254.01 | 1252.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 09:15:00 | 1261.95 | 1285.41 | 1270.35 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-11-19 09:15:00 | 1309.20 | 1282.21 | 1270.14 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 1276.00 | 1283.76 | 1271.29 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-11-21 09:15:00 | 1268.95 | 1283.61 | 1271.28 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 09:15:00 | 1198.05 | 1283.80 | 1284.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 11:15:00 | 1193.00 | 1282.04 | 1283.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 09:15:00 | 809.45 | 766.41 | 874.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-30 10:15:00 | 705.50 | 814.15 | 845.49 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 623.85 | 602.01 | 644.25 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-28 12:15:00 | 592.40 | 604.14 | 639.57 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 620.10 | 598.59 | 623.26 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-19 14:15:00 | 611.65 | 602.26 | 622.42 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 599.30 | 599.42 | 617.52 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-29 10:15:00 | 594.20 | 599.36 | 617.40 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-27 09:15:00 | 407.90 | 348.10 | 385.38 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 14:15:00 | 451.35 | 411.09 | 410.92 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-08-26 14:15:00 | 1262.00 | 2024-08-27 13:15:00 | 1310.05 | EXIT_EMA400 | -48.05 |
| SELL | 2024-08-27 09:15:00 | 1251.10 | 2024-08-27 13:15:00 | 1310.05 | EXIT_EMA400 | -58.95 |
| SELL | 2024-10-21 11:15:00 | 1276.05 | 2024-10-21 12:15:00 | 1299.05 | EXIT_EMA400 | -23.00 |
| BUY | 2024-11-19 09:15:00 | 1309.20 | 2024-11-21 09:15:00 | 1268.95 | EXIT_EMA400 | -40.25 |
| SELL | 2025-09-19 14:15:00 | 611.65 | 2025-09-26 09:15:00 | 579.35 | TARGET | 32.30 |
| SELL | 2025-09-29 10:15:00 | 594.20 | 2025-11-07 09:15:00 | 524.59 | TARGET | 69.61 |
| SELL | 2025-08-28 12:15:00 | 592.40 | 2025-12-18 09:15:00 | 450.88 | TARGET | 141.52 |
| SELL | 2025-04-30 10:15:00 | 705.50 | 2026-02-27 09:15:00 | 407.90 | EXIT_EMA400 | 297.60 |
