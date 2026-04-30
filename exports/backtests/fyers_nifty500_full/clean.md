# Clean Science and Technology Ltd. (CLEAN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 820.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** -29.14
- **Avg P&L per closed trade:** -7.28

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 1468.95 | 1531.54 | 1531.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 1435.95 | 1529.53 | 1530.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 11:15:00 | 1378.50 | 1359.73 | 1418.66 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-27 09:15:00 | 1339.00 | 1418.65 | 1425.48 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 1404.75 | 1401.52 | 1415.35 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-31 14:15:00 | 1422.00 | 1402.07 | 1415.29 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 14:15:00 | 1425.20 | 1255.50 | 1254.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 10:15:00 | 1441.50 | 1271.93 | 1263.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 1407.20 | 1412.68 | 1360.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 09:15:00 | 1444.80 | 1413.64 | 1363.07 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-18 09:15:00 | 1350.60 | 1444.01 | 1407.19 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 10:15:00 | 1233.90 | 1379.04 | 1379.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 13:15:00 | 1224.40 | 1357.26 | 1367.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 09:15:00 | 1195.70 | 1192.79 | 1239.04 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-19 14:15:00 | 1178.00 | 1192.68 | 1234.80 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 747.30 | 721.12 | 757.94 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-16 11:15:00 | 744.05 | 723.45 | 757.68 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-17 09:15:00 | 766.40 | 724.83 | 757.53 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-27 09:15:00 | 1339.00 | 2025-01-31 14:15:00 | 1422.00 | EXIT_EMA400 | -83.00 |
| BUY | 2025-06-24 09:15:00 | 1444.80 | 2025-07-18 09:15:00 | 1350.60 | EXIT_EMA400 | -94.20 |
| SELL | 2025-09-19 14:15:00 | 1178.00 | 2025-10-31 15:15:00 | 1007.59 | TARGET | 170.41 |
| SELL | 2026-04-16 11:15:00 | 744.05 | 2026-04-17 09:15:00 | 766.40 | EXIT_EMA400 | -22.35 |
