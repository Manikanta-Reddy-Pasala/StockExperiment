# SBI Life Insurance Company Ltd. (SBILIFE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1820.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 5 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 1
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** -139.20
- **Avg P&L per closed trade:** -27.84

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 14:15:00 | 1604.95 | 1742.09 | 1742.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 1593.40 | 1716.00 | 1728.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 11:15:00 | 1448.95 | 1446.36 | 1514.41 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 14:15:00 | 1429.95 | 1445.87 | 1510.84 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 1499.70 | 1455.18 | 1503.58 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-14 15:15:00 | 1492.15 | 1455.55 | 1503.52 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 1472.90 | 1455.72 | 1503.37 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-16 09:15:00 | 1538.60 | 1457.75 | 1502.76 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 09:15:00 | 1538.50 | 1473.16 | 1472.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 13:15:00 | 1551.00 | 1475.89 | 1474.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1464.95 | 1496.14 | 1485.79 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-08 14:15:00 | 1487.90 | 1492.65 | 1484.58 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 1487.90 | 1492.65 | 1484.58 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-04-08 15:15:00 | 1493.50 | 1492.66 | 1484.63 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 1488.50 | 1492.62 | 1484.65 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-04-09 14:15:00 | 1481.85 | 1492.33 | 1484.70 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 1786.70 | 1809.18 | 1809.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 1778.00 | 1808.87 | 1809.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 13:15:00 | 1808.20 | 1805.73 | 1807.44 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 1850.00 | 1809.08 | 1809.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 1898.00 | 1821.18 | 1815.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 14:15:00 | 1963.70 | 1964.61 | 1916.05 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-01 15:15:00 | 1974.00 | 1964.58 | 1917.94 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 2023.80 | 2046.62 | 2010.05 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-23 09:15:00 | 1991.60 | 2045.86 | 2010.03 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 15:15:00 | 1963.00 | 2013.11 | 2013.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 09:15:00 | 1940.10 | 2012.38 | 2012.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1914.90 | 1898.14 | 1942.45 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-22 10:15:00 | 1884.70 | 1918.68 | 1942.19 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-06 14:15:00 | 1429.95 | 2025-01-16 09:15:00 | 1538.60 | EXIT_EMA400 | -108.65 |
| SELL | 2025-01-14 15:15:00 | 1492.15 | 2025-01-16 09:15:00 | 1538.60 | EXIT_EMA400 | -46.45 |
| BUY | 2025-04-08 14:15:00 | 1487.90 | 2025-04-09 09:15:00 | 1497.85 | TARGET | 9.95 |
| BUY | 2025-04-08 15:15:00 | 1493.50 | 2025-04-09 14:15:00 | 1481.85 | EXIT_EMA400 | -11.65 |
| BUY | 2025-12-01 15:15:00 | 1974.00 | 2026-01-23 09:15:00 | 1991.60 | EXIT_EMA400 | 17.60 |
