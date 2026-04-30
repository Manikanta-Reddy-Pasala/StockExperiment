# Bharti Hexacom Ltd. (BHARTIHEXA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1517.00
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
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 3 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -55.78
- **Avg P&L per closed trade:** -9.30

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 15:15:00 | 1270.00 | 1402.56 | 1403.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 1268.05 | 1389.95 | 1396.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 10:15:00 | 1386.10 | 1377.06 | 1388.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-30 14:15:00 | 1355.95 | 1376.69 | 1388.49 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 1358.10 | 1375.09 | 1387.15 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-03 09:15:00 | 1336.60 | 1374.13 | 1386.25 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-02-03 15:15:00 | 1408.40 | 1373.72 | 1385.68 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 1471.70 | 1362.21 | 1362.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 15:15:00 | 1476.00 | 1366.18 | 1364.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 11:15:00 | 1742.20 | 1751.87 | 1666.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-18 09:15:00 | 1775.20 | 1751.52 | 1668.72 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-14 09:15:00 | 1743.90 | 1819.12 | 1752.00 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 09:15:00 | 1716.30 | 1766.45 | 1766.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 10:15:00 | 1699.50 | 1765.78 | 1766.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 10:15:00 | 1744.60 | 1708.98 | 1731.67 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 14:15:00 | 1791.00 | 1744.91 | 1744.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 1826.60 | 1746.10 | 1745.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 1765.10 | 1791.71 | 1771.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-11 09:15:00 | 1803.20 | 1787.73 | 1771.06 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1803.20 | 1787.73 | 1771.06 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-11-13 09:15:00 | 1815.20 | 1789.14 | 1772.91 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-11-14 09:15:00 | 1767.20 | 1790.05 | 1773.94 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 12:15:00 | 1709.40 | 1769.86 | 1770.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 13:15:00 | 1697.80 | 1769.14 | 1769.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 12:15:00 | 1749.50 | 1746.64 | 1756.93 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 09:15:00 | 1848.00 | 1764.40 | 1764.38 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 12:15:00 | 1680.00 | 1767.29 | 1767.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 09:15:00 | 1657.60 | 1763.65 | 1765.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 09:15:00 | 1701.40 | 1652.03 | 1693.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-27 11:15:00 | 1626.50 | 1668.62 | 1687.76 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-30 14:15:00 | 1355.95 | 2025-02-03 15:15:00 | 1408.40 | EXIT_EMA400 | -52.45 |
| SELL | 2025-02-03 09:15:00 | 1336.60 | 2025-02-03 15:15:00 | 1408.40 | EXIT_EMA400 | -71.80 |
| BUY | 2025-06-18 09:15:00 | 1775.20 | 2025-07-14 09:15:00 | 1743.90 | EXIT_EMA400 | -31.30 |
| BUY | 2025-11-11 09:15:00 | 1803.20 | 2025-11-14 09:15:00 | 1767.20 | EXIT_EMA400 | -36.00 |
| BUY | 2025-11-13 09:15:00 | 1815.20 | 2025-11-14 09:15:00 | 1767.20 | EXIT_EMA400 | -48.00 |
| SELL | 2026-02-27 11:15:00 | 1626.50 | 2026-04-02 09:15:00 | 1442.73 | TARGET | 183.77 |
