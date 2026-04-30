# Home First Finance Company India Ltd. (HOMEFIRST.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1166.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 3 |
| EXIT | 6 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 4 / 5
- **Target hits / EMA400 exits:** 4 / 5
- **Total realized P&L (per unit):** 175.54
- **Avg P&L per closed trade:** 19.50

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 09:15:00 | 1049.50 | 1136.79 | 1137.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 09:15:00 | 1001.95 | 1112.88 | 1123.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 12:15:00 | 1114.85 | 1108.35 | 1120.29 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-03 14:15:00 | 1056.00 | 1107.77 | 1119.88 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-30 09:15:00 | 1086.00 | 1048.79 | 1077.19 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 1127.50 | 1007.78 | 1007.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 10:15:00 | 1138.50 | 1009.08 | 1007.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 10:15:00 | 1155.20 | 1157.11 | 1115.13 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-26 14:15:00 | 1175.70 | 1157.42 | 1118.89 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-29 09:15:00 | 1303.70 | 1363.51 | 1305.54 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 12:15:00 | 1255.20 | 1277.76 | 1277.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 13:15:00 | 1247.60 | 1277.46 | 1277.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1273.80 | 1272.75 | 1275.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-09 11:15:00 | 1255.30 | 1272.24 | 1274.73 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-10 09:15:00 | 1274.50 | 1271.65 | 1274.37 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 15:15:00 | 1295.10 | 1276.74 | 1276.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 1307.60 | 1277.05 | 1276.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 10:15:00 | 1274.30 | 1278.27 | 1277.50 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-18 09:15:00 | 1300.70 | 1278.06 | 1277.42 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 1277.80 | 1278.31 | 1277.56 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-19 09:15:00 | 1269.00 | 1278.26 | 1277.54 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 13:15:00 | 1264.70 | 1276.80 | 1276.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 1264.60 | 1276.68 | 1276.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 11:15:00 | 1277.50 | 1275.60 | 1276.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-25 09:15:00 | 1266.00 | 1275.72 | 1276.25 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1266.00 | 1275.72 | 1276.25 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-25 11:15:00 | 1248.20 | 1275.24 | 1276.01 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-15 10:15:00 | 1260.00 | 1241.84 | 1255.20 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 15:15:00 | 1187.60 | 1144.97 | 1144.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 11:15:00 | 1190.00 | 1146.06 | 1145.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 15:15:00 | 1153.90 | 1154.32 | 1150.05 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-20 12:15:00 | 1162.80 | 1154.26 | 1150.10 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 1162.80 | 1154.26 | 1150.10 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-02-20 13:15:00 | 1172.30 | 1154.44 | 1150.21 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 1170.80 | 1155.90 | 1151.17 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-02-24 10:15:00 | 1188.40 | 1156.23 | 1151.35 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-02-27 09:15:00 | 1151.10 | 1164.62 | 1156.27 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 12:15:00 | 1043.00 | 1148.52 | 1148.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 13:15:00 | 1038.50 | 1147.43 | 1148.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1060.60 | 1011.42 | 1059.22 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 1129.65 | 1084.81 | 1084.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 12:15:00 | 1151.60 | 1086.53 | 1085.59 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-03 14:15:00 | 1056.00 | 2024-12-30 09:15:00 | 1086.00 | EXIT_EMA400 | -30.00 |
| BUY | 2025-05-26 14:15:00 | 1175.70 | 2025-06-24 09:15:00 | 1346.13 | TARGET | 170.43 |
| SELL | 2025-09-09 11:15:00 | 1255.30 | 2025-09-10 09:15:00 | 1274.50 | EXIT_EMA400 | -19.20 |
| BUY | 2025-09-18 09:15:00 | 1300.70 | 2025-09-19 09:15:00 | 1269.00 | EXIT_EMA400 | -31.70 |
| SELL | 2025-09-25 09:15:00 | 1266.00 | 2025-09-25 14:15:00 | 1235.25 | TARGET | 30.75 |
| SELL | 2025-09-25 11:15:00 | 1248.20 | 2025-10-15 10:15:00 | 1260.00 | EXIT_EMA400 | -11.80 |
| BUY | 2026-02-20 12:15:00 | 1162.80 | 2026-02-24 12:15:00 | 1200.89 | TARGET | 38.09 |
| BUY | 2026-02-20 13:15:00 | 1172.30 | 2026-02-25 09:15:00 | 1238.56 | TARGET | 66.26 |
| BUY | 2026-02-24 10:15:00 | 1188.40 | 2026-02-27 09:15:00 | 1151.10 | EXIT_EMA400 | -37.30 |
