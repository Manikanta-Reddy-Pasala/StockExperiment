# KPIT Technologies Ltd. (KPITTECH.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 758.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 1 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** 108.17
- **Avg P&L per closed trade:** 21.63

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 09:15:00 | 1671.95 | 1720.71 | 1720.94 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 10:15:00 | 1754.25 | 1720.78 | 1720.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 11:15:00 | 1773.00 | 1721.30 | 1721.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 11:15:00 | 1737.30 | 1741.06 | 1732.00 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 12:15:00 | 1428.05 | 1722.79 | 1723.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 14:15:00 | 1407.95 | 1716.60 | 1720.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 1439.90 | 1436.78 | 1520.23 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-23 12:15:00 | 1410.00 | 1480.56 | 1516.20 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-30 09:15:00 | 1466.55 | 1375.43 | 1427.69 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 1321.50 | 1284.70 | 1284.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 15:15:00 | 1326.00 | 1286.06 | 1285.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 1324.40 | 1350.04 | 1326.52 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 12:15:00 | 1270.50 | 1310.29 | 1310.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 1263.60 | 1308.23 | 1309.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 10:15:00 | 1295.50 | 1295.30 | 1301.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-16 13:15:00 | 1291.60 | 1295.31 | 1301.76 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1226.50 | 1220.89 | 1242.17 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-08 09:15:00 | 1220.20 | 1221.38 | 1241.68 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-09 09:15:00 | 1243.00 | 1221.92 | 1241.26 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 14:15:00 | 1259.80 | 1206.27 | 1206.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 09:15:00 | 1283.20 | 1207.55 | 1206.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-09 09:15:00 | 1206.30 | 1216.00 | 1211.46 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-11 11:15:00 | 1228.70 | 1214.87 | 1211.22 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-15 13:15:00 | 1211.30 | 1217.04 | 1212.63 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 14:15:00 | 1162.40 | 1208.37 | 1208.56 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 09:15:00 | 1230.50 | 1208.66 | 1208.65 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 13:15:00 | 1168.10 | 1208.65 | 1208.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 14:15:00 | 1160.60 | 1208.18 | 1208.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 1193.60 | 1193.21 | 1200.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-20 10:15:00 | 1136.90 | 1186.75 | 1195.05 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-23 12:15:00 | 1410.00 | 2025-01-30 09:15:00 | 1466.55 | EXIT_EMA400 | -56.55 |
| SELL | 2025-07-16 13:15:00 | 1291.60 | 2025-07-21 09:15:00 | 1261.12 | TARGET | 30.48 |
| SELL | 2025-09-08 09:15:00 | 1220.20 | 2025-09-09 09:15:00 | 1243.00 | EXIT_EMA400 | -22.80 |
| BUY | 2025-12-11 11:15:00 | 1228.70 | 2025-12-15 13:15:00 | 1211.30 | EXIT_EMA400 | -17.40 |
| SELL | 2026-01-20 10:15:00 | 1136.90 | 2026-02-04 09:15:00 | 962.46 | TARGET | 174.44 |
