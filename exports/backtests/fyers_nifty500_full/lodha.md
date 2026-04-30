# Lodha Developers Ltd. (LODHA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 893.55
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 6 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 308.36
- **Avg P&L per closed trade:** 44.05

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 13:15:00 | 1184.00 | 1355.16 | 1355.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 11:15:00 | 1177.10 | 1309.80 | 1329.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 09:15:00 | 1285.05 | 1246.79 | 1281.96 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-30 13:15:00 | 1240.00 | 1295.05 | 1299.09 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 1217.00 | 1182.25 | 1221.95 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-01 18:15:00 | 1202.65 | 1182.45 | 1221.85 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1205.40 | 1182.68 | 1221.77 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-04 10:15:00 | 1197.55 | 1182.83 | 1221.65 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 1213.80 | 1182.52 | 1218.69 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-06 14:15:00 | 1220.95 | 1183.57 | 1218.68 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 15:15:00 | 1292.50 | 1233.35 | 1233.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 09:15:00 | 1306.00 | 1234.08 | 1233.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 1358.35 | 1370.32 | 1322.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-03 11:15:00 | 1397.15 | 1370.99 | 1327.77 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 1346.30 | 1371.10 | 1330.36 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-07 11:15:00 | 1325.50 | 1370.32 | 1330.38 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 12:15:00 | 1174.00 | 1303.32 | 1303.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 09:15:00 | 1168.55 | 1291.91 | 1297.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 11:15:00 | 1230.80 | 1219.08 | 1254.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-31 14:15:00 | 1205.15 | 1218.79 | 1253.44 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1237.45 | 1218.56 | 1252.47 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-02-01 13:15:00 | 1259.00 | 1218.97 | 1252.50 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 11:15:00 | 1352.60 | 1196.96 | 1196.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 13:15:00 | 1357.70 | 1200.09 | 1198.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 1241.80 | 1258.82 | 1233.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 1292.30 | 1257.75 | 1234.12 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 1385.00 | 1430.32 | 1382.23 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-01 09:15:00 | 1372.80 | 1429.75 | 1382.18 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 12:15:00 | 1244.30 | 1373.35 | 1373.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 13:15:00 | 1236.10 | 1371.99 | 1373.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 13:15:00 | 1299.80 | 1285.48 | 1318.48 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-22 10:15:00 | 1276.10 | 1286.70 | 1317.35 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-31 09:15:00 | 1202.80 | 1175.70 | 1202.02 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-30 13:15:00 | 1240.00 | 2024-10-23 09:15:00 | 1062.73 | TARGET | 177.27 |
| SELL | 2024-11-01 18:15:00 | 1202.65 | 2024-11-06 14:15:00 | 1220.95 | EXIT_EMA400 | -18.30 |
| SELL | 2024-11-04 10:15:00 | 1197.55 | 2024-11-06 14:15:00 | 1220.95 | EXIT_EMA400 | -23.40 |
| BUY | 2025-01-03 11:15:00 | 1397.15 | 2025-01-07 11:15:00 | 1325.50 | EXIT_EMA400 | -71.65 |
| SELL | 2025-01-31 14:15:00 | 1205.15 | 2025-02-01 13:15:00 | 1259.00 | EXIT_EMA400 | -53.85 |
| BUY | 2025-05-12 09:15:00 | 1292.30 | 2025-05-26 09:15:00 | 1466.84 | TARGET | 174.54 |
| SELL | 2025-08-22 10:15:00 | 1276.10 | 2025-09-25 13:15:00 | 1152.35 | TARGET | 123.75 |
