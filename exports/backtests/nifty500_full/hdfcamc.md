# HDFC Asset Management Company Ltd. (HDFCAMC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 2712.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** 337.16
- **Avg P&L per closed trade:** 67.43

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 13:15:00 | 2121.45 | 2172.20 | 2172.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 12:15:00 | 2110.60 | 2169.20 | 2170.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 2167.18 | 2158.54 | 2164.88 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 15:15:00 | 2229.82 | 2170.13 | 2170.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 09:15:00 | 2232.23 | 2170.75 | 2170.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 09:15:00 | 2189.60 | 2193.49 | 2182.91 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 15:15:00 | 2130.00 | 2174.83 | 2175.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 12:15:00 | 2120.10 | 2173.16 | 2174.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 10:15:00 | 2000.57 | 1993.92 | 2053.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-06 12:15:00 | 1976.07 | 1994.38 | 2050.84 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 10:15:00 | 1945.35 | 1901.77 | 1949.97 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-19 11:15:00 | 1954.57 | 1902.30 | 1949.99 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 2188.90 | 1975.53 | 1974.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 10:15:00 | 2202.05 | 1977.78 | 1976.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 2112.55 | 2116.45 | 2062.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 2223.75 | 2117.65 | 2065.05 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 2816.25 | 2855.60 | 2779.69 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-29 13:15:00 | 2823.75 | 2855.28 | 2779.91 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-09-30 10:15:00 | 2776.50 | 2853.03 | 2780.26 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 13:15:00 | 2740.50 | 2771.79 | 2771.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 2723.75 | 2768.12 | 2769.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 2674.90 | 2667.28 | 2705.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-15 09:15:00 | 2638.90 | 2667.18 | 2704.22 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-18 12:15:00 | 2716.00 | 2652.52 | 2692.23 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 13:15:00 | 2831.00 | 2645.37 | 2645.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 12:15:00 | 2841.70 | 2667.04 | 2656.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 11:15:00 | 2694.20 | 2702.61 | 2678.45 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-25 09:15:00 | 2716.10 | 2702.07 | 2678.78 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-02 09:15:00 | 2680.00 | 2707.65 | 2684.07 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 13:15:00 | 2437.10 | 2662.66 | 2663.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 2431.70 | 2634.44 | 2648.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2523.90 | 2445.21 | 2523.88 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 10:15:00 | 2705.10 | 2571.79 | 2571.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 14:15:00 | 2733.60 | 2577.38 | 2574.60 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-06 12:15:00 | 1976.07 | 2025-03-19 11:15:00 | 1954.57 | EXIT_EMA400 | 21.50 |
| BUY | 2025-05-12 09:15:00 | 2223.75 | 2025-07-17 13:15:00 | 2699.86 | TARGET | 476.11 |
| BUY | 2025-09-29 13:15:00 | 2823.75 | 2025-09-30 10:15:00 | 2776.50 | EXIT_EMA400 | -47.25 |
| SELL | 2025-12-15 09:15:00 | 2638.90 | 2025-12-18 12:15:00 | 2716.00 | EXIT_EMA400 | -77.10 |
| BUY | 2026-02-25 09:15:00 | 2716.10 | 2026-03-02 09:15:00 | 2680.00 | EXIT_EMA400 | -36.10 |
