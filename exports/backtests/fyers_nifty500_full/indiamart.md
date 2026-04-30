# Indiamart Intermesh Ltd. (INDIAMART.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 2097.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 2 / 1
- **Target hits / EMA400 exits:** 2 / 1
- **Total realized P&L (per unit):** 393.16
- **Avg P&L per closed trade:** 131.05

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 13:15:00 | 2490.80 | 2901.36 | 2901.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 11:15:00 | 2462.65 | 2880.70 | 2891.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 2358.25 | 2318.25 | 2417.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-10 09:15:00 | 2295.00 | 2318.94 | 2414.40 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-20 13:15:00 | 2129.50 | 2043.07 | 2124.59 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 13:15:00 | 2225.00 | 2132.90 | 2132.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 15:15:00 | 2239.70 | 2134.94 | 2133.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 2557.30 | 2561.96 | 2471.22 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-24 09:15:00 | 2626.90 | 2563.73 | 2478.24 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-05 09:15:00 | 2467.50 | 2567.16 | 2501.35 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 14:15:00 | 2375.00 | 2531.32 | 2531.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 15:15:00 | 2360.00 | 2529.61 | 2530.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 10:15:00 | 2439.00 | 2411.09 | 2452.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-20 13:15:00 | 2388.70 | 2440.26 | 2453.49 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-16 09:15:00 | 2282.80 | 2217.49 | 2281.34 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-10 09:15:00 | 2295.00 | 2025-02-27 15:15:00 | 1936.80 | TARGET | 358.20 |
| BUY | 2025-07-24 09:15:00 | 2626.90 | 2025-08-05 09:15:00 | 2467.50 | EXIT_EMA400 | -159.40 |
| SELL | 2025-11-20 13:15:00 | 2388.70 | 2025-12-29 14:15:00 | 2194.34 | TARGET | 194.36 |
