# Info Edge (India) Ltd. (NAUKRI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 972.85
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
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** -56.70
- **Avg P&L per closed trade:** -11.34

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 11:15:00 | 853.97 | 879.42 | 879.43 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-09-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 13:15:00 | 896.38 | 879.22 | 879.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 14:15:00 | 899.10 | 879.42 | 879.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-13 10:15:00 | 874.45 | 883.65 | 881.57 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-09-15 13:15:00 | 891.20 | 883.41 | 881.61 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 10:15:00 | 888.01 | 885.24 | 882.73 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2023-09-21 12:15:00 | 882.65 | 885.21 | 882.74 | Close below EMA400 |

### Cycle 3 — SELL (started 2023-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 12:15:00 | 846.20 | 880.46 | 880.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 10:15:00 | 839.51 | 876.74 | 878.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-03 09:15:00 | 839.67 | 836.43 | 850.18 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2023-11-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 13:15:00 | 944.40 | 860.12 | 859.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 14:15:00 | 949.07 | 861.01 | 860.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 1005.64 | 1015.93 | 977.16 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-02 11:15:00 | 1038.72 | 1009.99 | 984.41 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-05 09:15:00 | 1021.95 | 1048.08 | 1024.43 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 11:15:00 | 1498.49 | 1628.91 | 1629.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 13:15:00 | 1491.00 | 1626.22 | 1628.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 09:15:00 | 1569.77 | 1563.47 | 1591.15 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-04 09:15:00 | 1539.89 | 1562.77 | 1589.84 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-05 11:15:00 | 1590.17 | 1561.87 | 1588.18 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 14:15:00 | 1491.00 | 1429.14 | 1429.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 12:15:00 | 1502.80 | 1435.79 | 1433.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 1452.00 | 1456.64 | 1444.83 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-13 11:15:00 | 1457.60 | 1456.54 | 1444.90 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1467.40 | 1478.05 | 1461.41 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-01 12:15:00 | 1454.90 | 1477.68 | 1461.39 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 13:15:00 | 1400.60 | 1451.58 | 1451.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 10:15:00 | 1394.50 | 1449.55 | 1450.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 1455.90 | 1434.27 | 1442.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-28 12:15:00 | 1403.30 | 1438.26 | 1443.46 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-21 10:15:00 | 1422.50 | 1383.16 | 1405.96 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-09-15 13:15:00 | 891.20 | 2023-09-21 12:15:00 | 882.65 | EXIT_EMA400 | -8.55 |
| BUY | 2024-02-02 11:15:00 | 1038.72 | 2024-03-05 09:15:00 | 1021.95 | EXIT_EMA400 | -16.77 |
| SELL | 2025-02-04 09:15:00 | 1539.89 | 2025-02-05 11:15:00 | 1590.17 | EXIT_EMA400 | -50.28 |
| BUY | 2025-06-13 11:15:00 | 1457.60 | 2025-06-17 09:15:00 | 1495.70 | TARGET | 38.10 |
| SELL | 2025-07-28 12:15:00 | 1403.30 | 2025-08-21 10:15:00 | 1422.50 | EXIT_EMA400 | -19.20 |
