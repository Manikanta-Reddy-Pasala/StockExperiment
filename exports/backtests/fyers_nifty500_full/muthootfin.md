# Muthoot Finance Ltd. (MUTHOOTFIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 3432.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** -60.93
- **Avg P&L per closed trade:** -15.23

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 13:15:00 | 1813.00 | 1918.41 | 1918.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 11:15:00 | 1805.35 | 1913.31 | 1916.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 11:15:00 | 1894.95 | 1890.97 | 1904.12 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 14:15:00 | 1938.00 | 1911.74 | 1911.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 09:15:00 | 1958.55 | 1912.48 | 1912.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 14:15:00 | 2097.00 | 2097.96 | 2037.96 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-14 10:15:00 | 2135.40 | 2098.39 | 2039.07 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 2161.65 | 2206.29 | 2153.29 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-02-24 12:15:00 | 2170.75 | 2205.93 | 2153.38 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-02-28 09:15:00 | 2144.95 | 2204.40 | 2157.10 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 13:15:00 | 2161.50 | 2200.33 | 2200.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 09:15:00 | 2136.10 | 2199.05 | 2199.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 2213.70 | 2197.49 | 2199.03 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 09:15:00 | 2297.10 | 2201.32 | 2200.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 10:15:00 | 2317.70 | 2202.48 | 2201.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 2205.30 | 2212.46 | 2206.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 2245.30 | 2213.35 | 2207.37 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 2211.40 | 2216.75 | 2209.42 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-13 13:15:00 | 2208.30 | 2216.67 | 2209.41 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 09:15:00 | 2090.40 | 2203.03 | 2203.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 15:15:00 | 2084.00 | 2196.36 | 2199.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 2150.20 | 2145.97 | 2170.10 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 15:15:00 | 2454.80 | 2189.47 | 2188.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 2508.50 | 2192.64 | 2190.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 10:15:00 | 2597.40 | 2606.95 | 2508.71 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-29 12:15:00 | 2625.10 | 2607.17 | 2509.79 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 2546.30 | 2611.49 | 2540.59 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-12 11:15:00 | 2539.50 | 2610.77 | 2540.59 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 3473.80 | 3671.04 | 3671.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 11:15:00 | 3444.40 | 3647.05 | 3659.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 3447.20 | 3329.04 | 3433.04 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-02-24 12:15:00 | 2170.75 | 2025-02-25 09:15:00 | 2222.87 | TARGET | 52.12 |
| BUY | 2025-01-14 10:15:00 | 2135.40 | 2025-02-28 09:15:00 | 2144.95 | EXIT_EMA400 | 9.55 |
| BUY | 2025-05-12 09:15:00 | 2245.30 | 2025-05-13 13:15:00 | 2208.30 | EXIT_EMA400 | -37.00 |
| BUY | 2025-07-29 12:15:00 | 2625.10 | 2025-08-12 11:15:00 | 2539.50 | EXIT_EMA400 | -85.60 |
