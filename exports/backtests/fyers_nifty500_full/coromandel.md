# Coromandel International Ltd. (COROMANDEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1994.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 6 |
| ENTRY1 | 6 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 1
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -205.30
- **Avg P&L per closed trade:** -25.66

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 15:15:00 | 1598.00 | 1634.43 | 1634.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 10:15:00 | 1581.80 | 1631.61 | 1633.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 11:15:00 | 1631.70 | 1629.87 | 1632.17 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 09:15:00 | 1655.00 | 1634.24 | 1634.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 13:15:00 | 1667.35 | 1634.98 | 1634.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 09:15:00 | 1733.95 | 1733.96 | 1700.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-10 09:15:00 | 1776.20 | 1735.88 | 1702.73 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 14:15:00 | 1810.05 | 1857.10 | 1799.93 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-01-14 09:15:00 | 1837.35 | 1856.47 | 1800.18 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-01-22 09:15:00 | 1809.00 | 1855.15 | 1810.06 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 1688.00 | 1800.70 | 1800.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 12:15:00 | 1681.15 | 1791.62 | 1796.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 11:15:00 | 1761.45 | 1745.35 | 1769.79 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-05 12:15:00 | 1704.85 | 1744.95 | 1769.47 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 10:15:00 | 1751.00 | 1742.81 | 1766.92 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-10 11:15:00 | 1770.40 | 1743.96 | 1766.56 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 15:15:00 | 1982.15 | 1783.64 | 1782.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 09:15:00 | 1997.90 | 1785.77 | 1783.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 14:15:00 | 2309.40 | 2314.61 | 2183.32 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-30 09:15:00 | 2357.70 | 2315.00 | 2184.82 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-27 10:15:00 | 2246.80 | 2315.97 | 2251.20 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 11:15:00 | 2215.00 | 2346.75 | 2347.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 10:15:00 | 2204.40 | 2326.29 | 2336.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 14:15:00 | 2326.40 | 2314.14 | 2329.05 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-18 10:15:00 | 2293.40 | 2313.87 | 2328.70 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 2272.50 | 2312.95 | 2327.79 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-19 10:15:00 | 2236.50 | 2312.19 | 2327.34 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 2302.10 | 2279.79 | 2304.68 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-06 11:15:00 | 2305.50 | 2281.14 | 2304.39 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 09:15:00 | 2377.30 | 2253.78 | 2253.63 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 10:15:00 | 2275.40 | 2283.34 | 2283.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 2242.10 | 2282.85 | 2283.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 14:15:00 | 2290.00 | 2274.90 | 2278.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-04 09:15:00 | 2247.50 | 2274.67 | 2278.76 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 2247.50 | 2274.67 | 2278.76 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-04 10:15:00 | 2244.50 | 2274.37 | 2278.59 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 2274.00 | 2273.91 | 2278.27 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-05 11:15:00 | 2281.10 | 2273.98 | 2278.22 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 2358.00 | 2280.54 | 2280.19 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 2173.10 | 2280.06 | 2280.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 2103.40 | 2275.53 | 2278.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2064.30 | 2024.47 | 2110.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-17 10:15:00 | 2032.10 | 2055.22 | 2110.98 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-12-10 09:15:00 | 1776.20 | 2025-01-22 09:15:00 | 1809.00 | EXIT_EMA400 | 32.80 |
| BUY | 2025-01-14 09:15:00 | 1837.35 | 2025-01-22 09:15:00 | 1809.00 | EXIT_EMA400 | -28.35 |
| SELL | 2025-03-05 12:15:00 | 1704.85 | 2025-03-10 11:15:00 | 1770.40 | EXIT_EMA400 | -65.55 |
| BUY | 2025-05-30 09:15:00 | 2357.70 | 2025-06-27 10:15:00 | 2246.80 | EXIT_EMA400 | -110.90 |
| SELL | 2025-09-18 10:15:00 | 2293.40 | 2025-09-26 14:15:00 | 2187.50 | TARGET | 105.90 |
| SELL | 2025-09-19 10:15:00 | 2236.50 | 2025-10-06 11:15:00 | 2305.50 | EXIT_EMA400 | -69.00 |
| SELL | 2026-02-04 09:15:00 | 2247.50 | 2026-02-05 11:15:00 | 2281.10 | EXIT_EMA400 | -33.60 |
| SELL | 2026-02-04 10:15:00 | 2244.50 | 2026-02-05 11:15:00 | 2281.10 | EXIT_EMA400 | -36.60 |
