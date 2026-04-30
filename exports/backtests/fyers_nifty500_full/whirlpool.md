# Whirlpool of India Ltd. (WHIRLPOOL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 985.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 7 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 4 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 321.41
- **Avg P&L per closed trade:** 45.92

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 15:15:00 | 1788.65 | 2129.01 | 2129.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 1749.75 | 2125.24 | 2127.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 14:15:00 | 1929.20 | 1928.69 | 1995.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-11 09:15:00 | 1899.00 | 1928.37 | 1994.50 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 1976.95 | 1928.87 | 1979.98 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-20 13:15:00 | 1934.85 | 1929.58 | 1979.58 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 1138.90 | 1061.91 | 1145.26 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-24 13:15:00 | 1161.50 | 1064.31 | 1145.22 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 1304.20 | 1188.53 | 1188.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 1357.00 | 1232.76 | 1216.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 12:15:00 | 1339.40 | 1341.08 | 1296.57 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-09 09:15:00 | 1362.10 | 1341.06 | 1297.44 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 1343.10 | 1379.09 | 1340.02 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-31 13:15:00 | 1340.00 | 1378.05 | 1340.08 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1287.00 | 1319.14 | 1319.22 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 10:15:00 | 1369.60 | 1318.68 | 1318.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 1378.00 | 1323.82 | 1321.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 14:15:00 | 1328.50 | 1332.85 | 1326.46 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-12 09:15:00 | 1338.50 | 1332.91 | 1326.55 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1338.50 | 1332.91 | 1326.55 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-12 10:15:00 | 1354.00 | 1333.12 | 1326.69 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1331.30 | 1337.33 | 1329.86 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-18 13:15:00 | 1326.70 | 1337.22 | 1329.84 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 1233.50 | 1324.06 | 1324.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 10:15:00 | 1222.10 | 1321.18 | 1322.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 14:15:00 | 1239.70 | 1238.29 | 1271.93 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 10:15:00 | 1407.30 | 1297.64 | 1297.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 14:15:00 | 1423.00 | 1302.06 | 1299.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 1314.00 | 1320.43 | 1309.87 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 1252.20 | 1301.47 | 1301.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 15:15:00 | 1241.30 | 1298.69 | 1300.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 10:15:00 | 839.85 | 838.07 | 924.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-13 11:15:00 | 832.30 | 886.00 | 910.27 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 847.15 | 837.95 | 870.58 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-09 09:15:00 | 830.00 | 838.30 | 869.64 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 858.15 | 837.88 | 864.32 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-17 12:15:00 | 874.00 | 838.61 | 864.29 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 14:15:00 | 996.50 | 884.31 | 883.77 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-20 13:15:00 | 1934.85 | 2024-12-30 14:15:00 | 1800.66 | TARGET | 134.19 |
| SELL | 2024-12-11 09:15:00 | 1899.00 | 2025-01-13 12:15:00 | 1612.51 | TARGET | 286.49 |
| BUY | 2025-07-09 09:15:00 | 1362.10 | 2025-07-31 13:15:00 | 1340.00 | EXIT_EMA400 | -22.10 |
| BUY | 2025-09-12 09:15:00 | 1338.50 | 2025-09-16 12:15:00 | 1374.34 | TARGET | 35.84 |
| BUY | 2025-09-12 10:15:00 | 1354.00 | 2025-09-18 13:15:00 | 1326.70 | EXIT_EMA400 | -27.30 |
| SELL | 2026-03-13 11:15:00 | 832.30 | 2026-04-17 12:15:00 | 874.00 | EXIT_EMA400 | -41.70 |
| SELL | 2026-04-09 09:15:00 | 830.00 | 2026-04-17 12:15:00 | 874.00 | EXIT_EMA400 | -44.00 |
