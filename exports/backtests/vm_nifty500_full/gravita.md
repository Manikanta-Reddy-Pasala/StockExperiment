# Gravita India Ltd. (GRAVITA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1630.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 4 |
| ENTRY1 | 8 |
| ENTRY2 | 1 |
| EXIT | 8 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / EMA400 exits:** 2 / 7
- **Total realized P&L (per unit):** -30.58
- **Avg P&L per closed trade:** -3.40

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 09:15:00 | 881.00 | 1019.05 | 1019.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 10:15:00 | 859.75 | 1017.47 | 1018.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-15 10:15:00 | 951.15 | 948.28 | 976.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-16 12:15:00 | 929.00 | 947.65 | 974.70 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 12:15:00 | 965.45 | 940.40 | 966.37 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-02-26 09:15:00 | 987.00 | 941.43 | 966.37 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-04-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 15:15:00 | 1085.15 | 943.07 | 942.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 09:15:00 | 1095.60 | 944.59 | 943.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-16 14:15:00 | 965.95 | 970.42 | 958.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-18 09:15:00 | 977.15 | 970.46 | 958.22 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 962.40 | 971.27 | 959.06 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-04-23 09:15:00 | 983.05 | 971.50 | 959.99 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-05-02 09:15:00 | 916.75 | 972.29 | 962.61 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-05-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-10 11:15:00 | 913.00 | 955.17 | 955.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 09:15:00 | 891.00 | 952.80 | 953.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-15 09:15:00 | 957.10 | 948.69 | 951.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-16 12:15:00 | 939.20 | 948.48 | 951.48 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-05-17 09:15:00 | 958.50 | 948.30 | 951.33 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 10:15:00 | 1027.85 | 954.10 | 954.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 11:15:00 | 1035.00 | 954.90 | 954.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 976.25 | 1012.89 | 987.66 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-05 14:15:00 | 1055.95 | 1012.79 | 988.94 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-22 11:15:00 | 2020.00 | 2401.70 | 2223.26 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 2130.00 | 2206.80 | 2207.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 12:15:00 | 2081.90 | 2202.36 | 2204.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 12:15:00 | 1754.55 | 1739.47 | 1863.08 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-04 12:15:00 | 1720.70 | 1762.64 | 1840.24 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 12:15:00 | 1772.95 | 1731.78 | 1813.47 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-11 14:15:00 | 1853.00 | 1733.46 | 1813.51 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 11:15:00 | 2010.00 | 1856.45 | 1855.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 12:15:00 | 2021.90 | 1858.10 | 1856.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 13:15:00 | 1917.10 | 1921.38 | 1893.55 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 11:15:00 | 1764.30 | 1877.76 | 1878.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 12:15:00 | 1753.20 | 1876.53 | 1877.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 11:15:00 | 1803.20 | 1801.28 | 1833.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-10 09:15:00 | 1780.00 | 1818.61 | 1834.75 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-07-24 09:15:00 | 1809.00 | 1773.96 | 1804.02 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 11:15:00 | 1754.50 | 1683.88 | 1683.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 15:15:00 | 1759.70 | 1686.71 | 1685.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 15:15:00 | 1817.80 | 1817.87 | 1780.67 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-06 09:15:00 | 1822.90 | 1817.92 | 1780.88 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 1784.50 | 1816.42 | 1782.45 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-08 09:15:00 | 1755.20 | 1815.81 | 1782.32 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 1562.00 | 1754.49 | 1755.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 1547.80 | 1737.23 | 1746.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 10:15:00 | 1674.90 | 1656.38 | 1697.05 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-18 14:15:00 | 1618.00 | 1658.67 | 1685.13 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-09 11:15:00 | 1539.50 | 1454.18 | 1527.06 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-02-16 12:15:00 | 929.00 | 2024-02-26 09:15:00 | 987.00 | EXIT_EMA400 | -58.00 |
| BUY | 2024-04-18 09:15:00 | 977.15 | 2024-05-02 09:15:00 | 916.75 | EXIT_EMA400 | -60.40 |
| BUY | 2024-04-23 09:15:00 | 983.05 | 2024-05-02 09:15:00 | 916.75 | EXIT_EMA400 | -66.30 |
| SELL | 2024-05-16 12:15:00 | 939.20 | 2024-05-17 09:15:00 | 958.50 | EXIT_EMA400 | -19.30 |
| BUY | 2024-06-05 14:15:00 | 1055.95 | 2024-06-12 10:15:00 | 1256.98 | TARGET | 201.03 |
| SELL | 2025-04-04 12:15:00 | 1720.70 | 2025-04-11 14:15:00 | 1853.00 | EXIT_EMA400 | -132.30 |
| SELL | 2025-07-10 09:15:00 | 1780.00 | 2025-07-24 09:15:00 | 1809.00 | EXIT_EMA400 | -29.00 |
| BUY | 2026-01-06 09:15:00 | 1822.90 | 2026-01-08 09:15:00 | 1755.20 | EXIT_EMA400 | -67.70 |
| SELL | 2026-02-18 14:15:00 | 1618.00 | 2026-03-13 14:15:00 | 1416.62 | TARGET | 201.38 |
