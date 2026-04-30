# Bata India Ltd. (BATAINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 719.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 114.69
- **Avg P&L per closed trade:** 19.11

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 12:15:00 | 1412.60 | 1461.86 | 1462.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 10:15:00 | 1388.55 | 1440.70 | 1448.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 10:15:00 | 1424.30 | 1420.09 | 1434.45 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-22 10:15:00 | 1403.20 | 1424.79 | 1434.70 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 1350.00 | 1350.05 | 1381.39 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-29 11:15:00 | 1390.00 | 1353.97 | 1380.06 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 11:15:00 | 1428.10 | 1397.98 | 1397.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 13:15:00 | 1435.00 | 1398.67 | 1398.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 11:15:00 | 1392.80 | 1400.84 | 1399.36 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 15:15:00 | 1366.05 | 1397.91 | 1397.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 09:15:00 | 1364.20 | 1397.57 | 1397.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 10:15:00 | 1408.40 | 1386.21 | 1391.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-30 09:15:00 | 1372.20 | 1386.16 | 1391.44 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-30 15:15:00 | 1395.45 | 1385.95 | 1391.18 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 10:15:00 | 1425.00 | 1395.41 | 1395.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 11:15:00 | 1429.90 | 1395.76 | 1395.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 11:15:00 | 1392.75 | 1397.04 | 1396.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-08 13:15:00 | 1412.85 | 1397.13 | 1396.16 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 13:15:00 | 1412.85 | 1397.13 | 1396.16 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-01-08 14:15:00 | 1424.00 | 1397.40 | 1396.30 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-01-13 09:15:00 | 1388.35 | 1401.06 | 1398.29 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 15:15:00 | 1348.00 | 1395.49 | 1395.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 09:15:00 | 1319.05 | 1394.73 | 1395.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 12:15:00 | 1340.40 | 1326.91 | 1353.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-14 13:15:00 | 1307.95 | 1339.25 | 1353.60 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-24 12:15:00 | 1343.30 | 1321.25 | 1340.95 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 11:15:00 | 1263.80 | 1195.67 | 1195.34 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 1164.70 | 1198.00 | 1198.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 1158.90 | 1197.61 | 1197.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 09:15:00 | 932.45 | 891.36 | 936.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-12 09:15:00 | 888.65 | 893.77 | 934.96 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-21 09:15:00 | 771.70 | 717.34 | 763.41 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-22 10:15:00 | 1403.20 | 2024-11-05 09:15:00 | 1308.70 | TARGET | 94.50 |
| SELL | 2024-12-30 09:15:00 | 1372.20 | 2024-12-30 15:15:00 | 1395.45 | EXIT_EMA400 | -23.25 |
| BUY | 2025-01-08 13:15:00 | 1412.85 | 2025-01-13 09:15:00 | 1388.35 | EXIT_EMA400 | -24.50 |
| BUY | 2025-01-08 14:15:00 | 1424.00 | 2025-01-13 09:15:00 | 1388.35 | EXIT_EMA400 | -35.65 |
| SELL | 2025-02-14 13:15:00 | 1307.95 | 2025-02-24 12:15:00 | 1343.30 | EXIT_EMA400 | -35.35 |
| SELL | 2026-02-12 09:15:00 | 888.65 | 2026-03-04 10:15:00 | 749.71 | TARGET | 138.94 |
