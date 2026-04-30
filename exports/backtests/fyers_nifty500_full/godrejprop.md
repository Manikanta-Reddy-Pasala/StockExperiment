# Godrej Properties Ltd. (GODREJPROP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1832.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / EMA400 exits:** 3 / 3
- **Total realized P&L (per unit):** 824.40
- **Avg P&L per closed trade:** 137.40

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 09:15:00 | 2900.00 | 2998.59 | 2998.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 10:15:00 | 2861.90 | 2983.67 | 2991.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 09:15:00 | 2967.90 | 2929.05 | 2956.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-16 12:15:00 | 2860.20 | 2929.43 | 2955.40 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-09-20 13:15:00 | 2954.15 | 2915.88 | 2944.49 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 14:15:00 | 3335.15 | 2970.26 | 2969.91 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 12:15:00 | 2853.25 | 3000.19 | 3000.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 2808.70 | 2992.34 | 2996.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 2830.45 | 2828.74 | 2898.34 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 10:15:00 | 2662.00 | 2846.81 | 2872.51 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-23 09:15:00 | 2146.80 | 2056.52 | 2142.56 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 11:15:00 | 2234.00 | 2157.84 | 2157.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 12:15:00 | 2248.00 | 2158.73 | 2157.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 15:15:00 | 2339.00 | 2342.41 | 2282.00 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-18 09:15:00 | 2351.40 | 2304.96 | 2281.47 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-25 09:15:00 | 2259.80 | 2315.93 | 2291.41 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 2101.40 | 2270.00 | 2270.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 2083.00 | 2263.29 | 2267.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 12:15:00 | 2047.00 | 2032.86 | 2097.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-24 14:15:00 | 2018.60 | 2054.09 | 2095.15 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 2070.90 | 2036.94 | 2074.82 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-07 15:15:00 | 2075.00 | 2037.32 | 2074.82 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 11:15:00 | 2313.50 | 2097.01 | 2096.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 2329.00 | 2118.83 | 2107.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 14:15:00 | 2190.50 | 2192.59 | 2152.29 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 12:15:00 | 2073.00 | 2139.59 | 2139.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 13:15:00 | 2062.30 | 2138.82 | 2139.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 10:15:00 | 2063.00 | 2049.00 | 2080.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-09 09:15:00 | 2024.60 | 2066.13 | 2084.54 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 2024.60 | 2066.13 | 2084.54 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-09 12:15:00 | 1998.30 | 2064.22 | 2083.30 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1849.10 | 1786.01 | 1867.26 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-18 10:15:00 | 1867.50 | 1786.82 | 1867.26 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-16 12:15:00 | 2860.20 | 2024-09-20 13:15:00 | 2954.15 | EXIT_EMA400 | -93.95 |
| SELL | 2025-01-06 10:15:00 | 2662.00 | 2025-02-11 11:15:00 | 2030.48 | TARGET | 631.52 |
| BUY | 2025-07-18 09:15:00 | 2351.40 | 2025-07-25 09:15:00 | 2259.80 | EXIT_EMA400 | -91.60 |
| SELL | 2025-09-24 14:15:00 | 2018.60 | 2025-10-07 15:15:00 | 2075.00 | EXIT_EMA400 | -56.40 |
| SELL | 2026-01-09 09:15:00 | 2024.60 | 2026-01-19 09:15:00 | 1844.79 | TARGET | 179.81 |
| SELL | 2026-01-09 12:15:00 | 1998.30 | 2026-01-20 11:15:00 | 1743.29 | TARGET | 255.01 |
