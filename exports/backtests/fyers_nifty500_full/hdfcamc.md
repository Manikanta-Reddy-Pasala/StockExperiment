# HDFC Asset Management Company Ltd. (HDFCAMC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 2719.60
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
- **Total realized P&L (per unit):** 339.82
- **Avg P&L per closed trade:** 67.96

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 14:15:00 | 2121.28 | 2171.33 | 2171.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 12:15:00 | 2110.60 | 2168.89 | 2170.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 2167.30 | 2158.55 | 2164.51 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 14:15:00 | 2227.45 | 2169.49 | 2169.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 09:15:00 | 2232.23 | 2170.71 | 2170.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 09:15:00 | 2189.60 | 2193.37 | 2182.58 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 09:15:00 | 2134.00 | 2174.34 | 2174.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 2100.00 | 2172.33 | 2173.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 10:15:00 | 2000.58 | 1990.95 | 2049.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-06 12:15:00 | 1976.08 | 1991.64 | 2047.41 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 10:15:00 | 1945.35 | 1901.16 | 1948.50 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-19 11:15:00 | 1954.58 | 1901.69 | 1948.53 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 15:15:00 | 2100.00 | 1973.21 | 1973.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 09:15:00 | 2188.90 | 1975.36 | 1974.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 2112.55 | 2116.29 | 2062.05 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 2223.75 | 2117.57 | 2064.56 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 2816.25 | 2855.49 | 2779.57 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-29 13:15:00 | 2823.75 | 2855.17 | 2779.79 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-09-30 10:15:00 | 2776.50 | 2852.91 | 2780.14 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 13:15:00 | 2740.50 | 2771.79 | 2771.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 2723.75 | 2768.07 | 2769.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 15:15:00 | 2668.00 | 2667.34 | 2705.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-15 09:15:00 | 2638.90 | 2667.29 | 2704.29 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-18 12:15:00 | 2714.80 | 2652.70 | 2692.33 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 13:15:00 | 2831.00 | 2642.74 | 2642.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 14:15:00 | 2836.00 | 2644.67 | 2642.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 11:15:00 | 2694.20 | 2701.21 | 2676.17 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-25 09:15:00 | 2716.10 | 2700.83 | 2676.60 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-02 09:15:00 | 2680.00 | 2706.54 | 2682.06 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 13:15:00 | 2437.10 | 2661.85 | 2661.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 2431.70 | 2633.85 | 2647.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2523.90 | 2445.06 | 2523.03 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 12:15:00 | 2734.80 | 2572.32 | 2571.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 11:15:00 | 2749.00 | 2590.63 | 2581.00 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-06 12:15:00 | 1976.08 | 2025-03-19 11:15:00 | 1954.58 | EXIT_EMA400 | 21.50 |
| BUY | 2025-05-12 09:15:00 | 2223.75 | 2025-07-17 13:15:00 | 2701.32 | TARGET | 477.57 |
| BUY | 2025-09-29 13:15:00 | 2823.75 | 2025-09-30 10:15:00 | 2776.50 | EXIT_EMA400 | -47.25 |
| SELL | 2025-12-15 09:15:00 | 2638.90 | 2025-12-18 12:15:00 | 2714.80 | EXIT_EMA400 | -75.90 |
| BUY | 2026-02-25 09:15:00 | 2716.10 | 2026-03-02 09:15:00 | 2680.00 | EXIT_EMA400 | -36.10 |
