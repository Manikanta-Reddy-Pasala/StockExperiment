# Poly Medicure Ltd. (POLYMED.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1519.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 4 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 2 / 2
- **Total realized P&L (per unit):** 469.44
- **Avg P&L per closed trade:** 117.36

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 14:15:00 | 2407.40 | 2623.04 | 2623.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 09:15:00 | 2348.30 | 2618.07 | 2621.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 11:15:00 | 2530.00 | 2483.84 | 2542.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-06 10:15:00 | 2407.95 | 2480.41 | 2537.23 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 2342.55 | 2249.80 | 2358.12 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-07 14:15:00 | 2284.20 | 2252.56 | 2356.85 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 2316.85 | 2244.49 | 2323.82 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-24 09:15:00 | 2337.90 | 2245.85 | 2323.72 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 10:15:00 | 2585.30 | 2311.44 | 2311.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 15:15:00 | 2602.20 | 2366.39 | 2340.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 2450.30 | 2455.44 | 2394.42 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-21 09:15:00 | 2506.20 | 2438.66 | 2396.45 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 2408.90 | 2442.08 | 2401.93 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-26 09:15:00 | 2388.50 | 2440.85 | 2401.91 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 10:15:00 | 2243.60 | 2374.75 | 2375.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 12:15:00 | 2235.10 | 2372.09 | 2373.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 14:15:00 | 2233.10 | 2231.85 | 2284.80 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-15 12:15:00 | 2182.30 | 2228.26 | 2266.63 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 2051.00 | 2003.05 | 2088.89 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-08-25 12:15:00 | 2126.00 | 2008.20 | 2088.55 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-06 10:15:00 | 2407.95 | 2025-02-28 09:15:00 | 2020.11 | TARGET | 387.84 |
| SELL | 2025-03-07 14:15:00 | 2284.20 | 2025-03-24 09:15:00 | 2337.90 | EXIT_EMA400 | -53.70 |
| BUY | 2025-05-21 09:15:00 | 2506.20 | 2025-05-26 09:15:00 | 2388.50 | EXIT_EMA400 | -117.70 |
| SELL | 2025-07-15 12:15:00 | 2182.30 | 2025-08-01 09:15:00 | 1929.30 | TARGET | 253.00 |
