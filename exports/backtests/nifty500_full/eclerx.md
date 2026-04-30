# eClerx Services Ltd. (ECLERX.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1429.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 1
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** 560.56
- **Avg P&L per closed trade:** 112.11

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 11:15:00 | 804.70 | 842.25 | 842.41 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-12 12:15:00 | 866.12 | 840.26 | 840.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-12 13:15:00 | 868.00 | 840.53 | 840.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-23 09:15:00 | 966.45 | 969.83 | 924.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-11-06 14:15:00 | 1015.88 | 974.96 | 940.36 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 14:15:00 | 1267.20 | 1315.08 | 1264.50 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-02-09 09:15:00 | 1262.03 | 1314.09 | 1264.51 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 10:15:00 | 1175.00 | 1239.36 | 1239.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 11:15:00 | 1161.65 | 1238.59 | 1239.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-15 15:15:00 | 1232.50 | 1231.38 | 1235.23 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-19 10:15:00 | 1195.57 | 1230.53 | 1234.62 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 13:15:00 | 1220.78 | 1212.16 | 1222.63 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-04-03 14:15:00 | 1224.85 | 1212.29 | 1222.64 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 10:15:00 | 1228.12 | 1183.87 | 1183.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 12:15:00 | 1232.00 | 1184.78 | 1184.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 14:15:00 | 1218.22 | 1221.59 | 1206.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-06 13:15:00 | 1239.05 | 1211.86 | 1205.63 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-08-14 09:15:00 | 1204.00 | 1225.78 | 1214.30 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 09:15:00 | 1460.38 | 1653.17 | 1654.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 1430.50 | 1563.27 | 1593.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 14:15:00 | 1393.43 | 1393.26 | 1458.34 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-04 09:15:00 | 1343.70 | 1392.74 | 1457.43 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-13 14:15:00 | 1350.50 | 1285.15 | 1346.30 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 15:15:00 | 1701.00 | 1394.51 | 1393.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 1736.40 | 1468.75 | 1433.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 1687.60 | 1692.16 | 1594.89 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-23 09:15:00 | 1731.20 | 1692.75 | 1596.15 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 2038.95 | 2117.98 | 2025.50 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-29 11:15:00 | 2016.50 | 2116.14 | 2025.50 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 13:15:00 | 2000.00 | 2252.05 | 2253.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 09:15:00 | 1958.75 | 2244.00 | 2249.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 09:15:00 | 1555.60 | 1553.51 | 1709.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-22 10:15:00 | 1484.30 | 1560.99 | 1695.58 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-11-06 14:15:00 | 1015.88 | 2023-11-22 12:15:00 | 1242.41 | TARGET | 226.54 |
| SELL | 2024-03-19 10:15:00 | 1195.57 | 2024-04-03 14:15:00 | 1224.85 | EXIT_EMA400 | -29.28 |
| BUY | 2024-08-06 13:15:00 | 1239.05 | 2024-08-14 09:15:00 | 1204.00 | EXIT_EMA400 | -35.05 |
| SELL | 2025-04-04 09:15:00 | 1343.70 | 2025-05-13 14:15:00 | 1350.50 | EXIT_EMA400 | -6.80 |
| BUY | 2025-06-23 09:15:00 | 1731.20 | 2025-08-25 14:15:00 | 2136.35 | TARGET | 405.15 |
