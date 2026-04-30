# Lloyds Metals And Energy Ltd. (LLOYDSME.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1759.00
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / EMA400 exits:** 0 / 3
- **Total realized P&L (per unit):** -84.20
- **Avg P&L per closed trade:** -28.07

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-03-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 09:15:00 | 990.85 | 1181.70 | 1182.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 10:15:00 | 972.80 | 1179.62 | 1181.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 09:15:00 | 1152.60 | 1142.72 | 1161.29 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2025-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 10:15:00 | 1306.50 | 1170.15 | 1170.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 10:15:00 | 1337.80 | 1210.67 | 1192.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1062.85 | 1223.28 | 1201.00 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-08 09:15:00 | 1165.70 | 1215.30 | 1197.70 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1165.70 | 1215.30 | 1197.70 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-04-08 10:15:00 | 1150.00 | 1214.65 | 1197.46 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 13:15:00 | 1355.40 | 1435.62 | 1435.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 1339.60 | 1434.67 | 1435.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 15:15:00 | 1315.00 | 1314.21 | 1351.08 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-06 12:15:00 | 1307.40 | 1314.29 | 1350.39 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 1345.80 | 1313.32 | 1346.44 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-09 12:15:00 | 1350.00 | 1313.69 | 1346.45 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 14:15:00 | 1388.30 | 1298.83 | 1298.44 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 15:15:00 | 1236.00 | 1300.75 | 1300.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 10:15:00 | 1228.90 | 1299.38 | 1300.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 1314.00 | 1201.33 | 1240.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-06 09:15:00 | 1225.80 | 1210.27 | 1242.78 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1225.80 | 1210.27 | 1242.78 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-06 10:15:00 | 1251.70 | 1210.69 | 1242.82 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 1351.00 | 1226.04 | 1225.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 15:15:00 | 1353.30 | 1227.31 | 1226.21 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-04-08 09:15:00 | 1165.70 | 2025-04-08 10:15:00 | 1150.00 | EXIT_EMA400 | -15.70 |
| SELL | 2025-10-06 12:15:00 | 1307.40 | 2025-10-09 12:15:00 | 1350.00 | EXIT_EMA400 | -42.60 |
| SELL | 2026-02-06 09:15:00 | 1225.80 | 2026-02-06 10:15:00 | 1251.70 | EXIT_EMA400 | -25.90 |
