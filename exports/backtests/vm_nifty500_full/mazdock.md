# Mazagoan Dock Shipbuilders Ltd. (MAZDOCK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 2733.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 0 / 7
- **Target hits / EMA400 exits:** 0 / 7
- **Total realized P&L (per unit):** -709.82
- **Avg P&L per closed trade:** -101.40

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 09:15:00 | 1035.00 | 1079.18 | 1079.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 1028.15 | 1071.53 | 1075.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 12:15:00 | 1004.67 | 995.19 | 1026.67 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-04-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 10:15:00 | 1098.95 | 1047.12 | 1047.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 12:15:00 | 1104.00 | 1048.21 | 1047.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 11:15:00 | 1101.90 | 1110.58 | 1084.71 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-08 09:15:00 | 1117.50 | 1109.70 | 1084.90 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 1117.50 | 1109.70 | 1084.90 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-05-08 12:15:00 | 1122.00 | 1109.91 | 1085.37 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-05-13 09:15:00 | 1069.80 | 1110.03 | 1087.58 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 13:15:00 | 2102.00 | 2177.00 | 2177.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 14:15:00 | 2098.43 | 2176.22 | 2176.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-10 10:15:00 | 2192.75 | 2132.05 | 2152.30 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 14:15:00 | 2337.52 | 2166.76 | 2166.63 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-10-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 14:15:00 | 2065.00 | 2166.22 | 2166.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 2036.28 | 2157.57 | 2161.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 2129.32 | 2109.60 | 2134.23 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-13 09:15:00 | 1974.43 | 2103.08 | 2126.80 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-26 09:15:00 | 2199.25 | 2069.34 | 2102.17 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 13:15:00 | 2327.50 | 2128.78 | 2128.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 09:15:00 | 2394.48 | 2135.79 | 2132.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-23 09:15:00 | 2320.90 | 2342.69 | 2258.90 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-20 11:15:00 | 2474.90 | 2266.75 | 2251.08 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-22 10:15:00 | 2248.45 | 2279.45 | 2258.69 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 14:15:00 | 2167.75 | 2257.11 | 2257.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-17 09:15:00 | 2088.00 | 2254.50 | 2255.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 2226.90 | 2187.35 | 2215.48 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 12:15:00 | 2377.65 | 2233.49 | 2233.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 2438.85 | 2239.63 | 2236.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 2282.00 | 2457.33 | 2369.50 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-08 09:15:00 | 2390.25 | 2448.19 | 2367.87 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-08 10:15:00 | 2362.00 | 2447.33 | 2367.84 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 11:15:00 | 2662.90 | 3092.31 | 3093.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 2639.00 | 2942.18 | 3007.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 14:15:00 | 2758.00 | 2757.06 | 2849.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-31 14:15:00 | 2730.60 | 2822.13 | 2846.01 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 2809.50 | 2775.35 | 2811.43 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-19 09:15:00 | 2775.00 | 2778.48 | 2810.61 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-11-20 09:15:00 | 2818.00 | 2778.83 | 2809.68 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 13:15:00 | 2661.30 | 2407.28 | 2407.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 09:15:00 | 2684.30 | 2414.98 | 2411.04 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-05-08 09:15:00 | 1117.50 | 2024-05-13 09:15:00 | 1069.80 | EXIT_EMA400 | -47.70 |
| BUY | 2024-05-08 12:15:00 | 1122.00 | 2024-05-13 09:15:00 | 1069.80 | EXIT_EMA400 | -52.20 |
| SELL | 2024-11-13 09:15:00 | 1974.43 | 2024-11-26 09:15:00 | 2199.25 | EXIT_EMA400 | -224.82 |
| BUY | 2025-01-20 11:15:00 | 2474.90 | 2025-01-22 10:15:00 | 2248.45 | EXIT_EMA400 | -226.45 |
| BUY | 2025-04-08 09:15:00 | 2390.25 | 2025-04-08 10:15:00 | 2362.00 | EXIT_EMA400 | -28.25 |
| SELL | 2025-10-31 14:15:00 | 2730.60 | 2025-11-20 09:15:00 | 2818.00 | EXIT_EMA400 | -87.40 |
| SELL | 2025-11-19 09:15:00 | 2775.00 | 2025-11-20 09:15:00 | 2818.00 | EXIT_EMA400 | -43.00 |
