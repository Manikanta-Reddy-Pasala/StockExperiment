# SBI Life Insurance Company Ltd. (SBILIFE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 1819.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 1
- **Winners / losers:** 4 / 1
- **Target hits / EMA400 exits:** 3 / 2
- **Total realized P&L (per unit):** 106.83
- **Avg P&L per closed trade:** 21.37

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 14:15:00 | 1460.85 | 1468.37 | 1468.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 11:15:00 | 1453.75 | 1468.08 | 1468.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 14:15:00 | 1453.15 | 1451.71 | 1458.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-21 09:15:00 | 1428.05 | 1450.59 | 1457.89 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 1452.00 | 1445.30 | 1453.60 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-05-29 09:15:00 | 1420.90 | 1445.05 | 1453.43 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-06-06 14:15:00 | 1443.50 | 1424.64 | 1440.28 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 12:15:00 | 1504.35 | 1447.54 | 1447.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 1511.45 | 1455.71 | 1451.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 11:15:00 | 1836.05 | 1838.72 | 1768.58 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 13:15:00 | 1604.45 | 1743.43 | 1743.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 1593.40 | 1717.71 | 1730.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 11:15:00 | 1448.85 | 1446.49 | 1514.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 14:15:00 | 1429.95 | 1445.99 | 1511.18 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 1472.40 | 1455.85 | 1503.67 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-16 09:15:00 | 1536.90 | 1457.83 | 1503.03 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 10:15:00 | 1543.25 | 1473.88 | 1473.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 13:15:00 | 1551.00 | 1475.90 | 1474.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1464.95 | 1496.19 | 1486.14 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-08 14:15:00 | 1487.90 | 1492.62 | 1484.88 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 1487.90 | 1492.62 | 1484.88 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-04-09 10:15:00 | 1484.70 | 1492.46 | 1484.91 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 1784.10 | 1809.06 | 1809.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 1777.90 | 1808.75 | 1809.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 13:15:00 | 1808.40 | 1805.63 | 1807.38 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 1850.00 | 1808.95 | 1808.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 1898.00 | 1821.14 | 1815.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 14:15:00 | 1963.70 | 1964.26 | 1915.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-01 15:15:00 | 1974.00 | 1964.25 | 1917.74 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 2023.80 | 2046.38 | 2009.85 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-23 09:15:00 | 1991.60 | 2045.60 | 2009.82 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 15:15:00 | 1963.70 | 2013.66 | 2013.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 09:15:00 | 1940.10 | 2012.93 | 2013.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1913.90 | 1898.43 | 1942.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-22 10:15:00 | 1884.80 | 1916.54 | 1941.65 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-05-21 09:15:00 | 1428.05 | 2024-06-04 11:15:00 | 1338.54 | TARGET | 89.51 |
| SELL | 2024-05-29 09:15:00 | 1420.90 | 2024-06-04 11:15:00 | 1323.30 | TARGET | 97.60 |
| SELL | 2025-01-06 14:15:00 | 1429.95 | 2025-01-16 09:15:00 | 1536.90 | EXIT_EMA400 | -106.95 |
| BUY | 2025-04-08 14:15:00 | 1487.90 | 2025-04-09 09:15:00 | 1496.97 | TARGET | 9.07 |
| BUY | 2025-12-01 15:15:00 | 1974.00 | 2026-01-23 09:15:00 | 1991.60 | EXIT_EMA400 | 17.60 |
