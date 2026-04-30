# Radico Khaitan Ltd (RADICO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 3435.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT3 | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** -106.17
- **Avg P&L per closed trade:** -35.39

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 11:15:00 | 2075.80 | 2317.17 | 2317.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 12:15:00 | 2048.15 | 2250.32 | 2277.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 14:15:00 | 2200.20 | 2185.88 | 2238.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-20 15:15:00 | 2139.00 | 2185.41 | 2238.29 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-06 14:15:00 | 2210.65 | 2133.07 | 2193.62 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 2357.00 | 2221.74 | 2221.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 14:15:00 | 2426.00 | 2227.30 | 2224.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 2233.45 | 2260.11 | 2242.42 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-08 09:15:00 | 2288.00 | 2259.29 | 2242.61 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 2545.90 | 2600.56 | 2533.75 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-02 12:15:00 | 2527.00 | 2598.58 | 2533.75 | Close below EMA400 |

### Cycle 3 — SELL (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 11:15:00 | 2933.90 | 3139.49 | 3140.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 14:15:00 | 2903.30 | 3133.01 | 3136.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 14:15:00 | 2772.70 | 2747.51 | 2853.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-09 09:15:00 | 2677.10 | 2747.18 | 2852.75 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-03-11 12:15:00 | 2847.80 | 2750.18 | 2845.65 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 11:15:00 | 3201.90 | 2821.63 | 2821.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 14:15:00 | 3255.00 | 2833.39 | 2827.33 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-20 15:15:00 | 2139.00 | 2025-03-06 14:15:00 | 2210.65 | EXIT_EMA400 | -71.65 |
| BUY | 2025-04-08 09:15:00 | 2288.00 | 2025-04-15 13:15:00 | 2424.18 | TARGET | 136.18 |
| SELL | 2026-03-09 09:15:00 | 2677.10 | 2026-03-11 12:15:00 | 2847.80 | EXIT_EMA400 | -170.70 |
