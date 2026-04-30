# Coromandel International Ltd. (COROMANDEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 1981.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 6 |
| ENTRY1 | 8 |
| ENTRY2 | 3 |
| EXIT | 7 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 1
- **Winners / losers:** 1 / 9
- **Target hits / EMA400 exits:** 1 / 9
- **Total realized P&L (per unit):** -380.40
- **Avg P&L per closed trade:** -38.04

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 09:15:00 | 1053.70 | 1168.12 | 1168.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 13:15:00 | 1029.90 | 1107.83 | 1128.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-05 09:15:00 | 1099.45 | 1097.88 | 1120.35 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-05 10:15:00 | 1092.10 | 1097.82 | 1120.21 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 1104.75 | 1097.12 | 1118.43 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-03-07 15:15:00 | 1122.00 | 1097.99 | 1118.24 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-04-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-15 15:15:00 | 1133.90 | 1115.01 | 1114.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-16 12:15:00 | 1139.10 | 1115.84 | 1115.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 09:15:00 | 1111.00 | 1117.77 | 1116.38 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-19 11:15:00 | 1119.05 | 1117.66 | 1116.33 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 11:15:00 | 1119.05 | 1117.66 | 1116.33 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-04-19 12:15:00 | 1121.35 | 1117.69 | 1116.36 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-04-22 09:15:00 | 1103.15 | 1117.67 | 1116.37 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 10:15:00 | 1104.65 | 1115.16 | 1115.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 11:15:00 | 1099.35 | 1115.01 | 1115.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 15:15:00 | 1133.00 | 1113.19 | 1114.14 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-04-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 11:15:00 | 1187.00 | 1115.22 | 1115.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-29 12:15:00 | 1198.80 | 1120.64 | 1117.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 1693.80 | 1702.34 | 1623.53 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-17 10:15:00 | 1736.05 | 1701.32 | 1640.37 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 1659.95 | 1698.24 | 1646.54 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-09-23 11:15:00 | 1670.00 | 1697.58 | 1646.72 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-09-25 09:15:00 | 1647.15 | 1694.06 | 1647.88 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 14:15:00 | 1600.85 | 1632.81 | 1632.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 09:15:00 | 1594.75 | 1632.11 | 1632.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 11:15:00 | 1631.70 | 1629.87 | 1631.42 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 12:15:00 | 1650.20 | 1632.94 | 1632.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 13:15:00 | 1667.35 | 1633.99 | 1633.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 10:15:00 | 1836.20 | 1858.64 | 1799.34 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-16 09:15:00 | 1874.05 | 1853.24 | 1802.13 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-22 09:15:00 | 1809.00 | 1855.17 | 1809.91 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 11:15:00 | 1691.70 | 1800.23 | 1800.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 12:15:00 | 1681.15 | 1792.15 | 1796.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 11:15:00 | 1761.45 | 1745.48 | 1769.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-05 12:15:00 | 1704.50 | 1745.07 | 1769.31 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 10:15:00 | 1751.00 | 1742.97 | 1766.80 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-10 11:15:00 | 1770.25 | 1744.10 | 1766.44 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 14:15:00 | 1980.05 | 1781.72 | 1781.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 09:15:00 | 1998.50 | 1785.82 | 1783.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 14:15:00 | 2307.00 | 2314.85 | 2183.38 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-30 09:15:00 | 2357.70 | 2315.23 | 2184.88 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-27 10:15:00 | 2247.00 | 2316.14 | 2251.29 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 11:15:00 | 2215.50 | 2346.84 | 2347.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 10:15:00 | 2204.20 | 2326.42 | 2336.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 14:15:00 | 2326.40 | 2314.25 | 2329.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-18 10:15:00 | 2293.50 | 2313.91 | 2328.78 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 2272.50 | 2312.99 | 2327.88 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-19 10:15:00 | 2236.50 | 2312.22 | 2327.42 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 2302.10 | 2279.52 | 2304.59 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-06 11:15:00 | 2305.50 | 2280.88 | 2304.30 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 09:15:00 | 2377.30 | 2253.76 | 2253.59 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 2212.80 | 2282.66 | 2282.99 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 10:15:00 | 2372.10 | 2280.84 | 2280.58 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 2172.80 | 2280.78 | 2281.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 2103.40 | 2276.22 | 2278.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2064.30 | 2024.68 | 2110.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-17 10:15:00 | 2032.10 | 2055.33 | 2111.25 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-03-05 10:15:00 | 1092.10 | 2024-03-07 15:15:00 | 1122.00 | EXIT_EMA400 | -29.90 |
| BUY | 2024-04-19 11:15:00 | 1119.05 | 2024-04-22 09:15:00 | 1103.15 | EXIT_EMA400 | -15.90 |
| BUY | 2024-04-19 12:15:00 | 1121.35 | 2024-04-22 09:15:00 | 1103.15 | EXIT_EMA400 | -18.20 |
| BUY | 2024-09-17 10:15:00 | 1736.05 | 2024-09-25 09:15:00 | 1647.15 | EXIT_EMA400 | -88.90 |
| BUY | 2024-09-23 11:15:00 | 1670.00 | 2024-09-25 09:15:00 | 1647.15 | EXIT_EMA400 | -22.85 |
| BUY | 2025-01-16 09:15:00 | 1874.05 | 2025-01-22 09:15:00 | 1809.00 | EXIT_EMA400 | -65.05 |
| SELL | 2025-03-05 12:15:00 | 1704.50 | 2025-03-10 11:15:00 | 1770.25 | EXIT_EMA400 | -65.75 |
| BUY | 2025-05-30 09:15:00 | 2357.70 | 2025-06-27 10:15:00 | 2247.00 | EXIT_EMA400 | -110.70 |
| SELL | 2025-09-18 10:15:00 | 2293.50 | 2025-09-26 14:15:00 | 2187.65 | TARGET | 105.85 |
| SELL | 2025-09-19 10:15:00 | 2236.50 | 2025-10-06 11:15:00 | 2305.50 | EXIT_EMA400 | -69.00 |
