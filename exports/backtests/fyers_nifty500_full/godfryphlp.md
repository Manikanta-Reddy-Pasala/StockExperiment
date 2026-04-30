# Godfrey Phillips India Ltd. (GODFRYPHLP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 2255.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 3 / 1
- **Target hits / EMA400 exits:** 3 / 1
- **Total realized P&L (per unit):** 1092.89
- **Avg P&L per closed trade:** 273.22

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 11:15:00 | 1922.65 | 2084.40 | 2085.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-03 09:15:00 | 1897.22 | 2076.21 | 2080.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 10:15:00 | 2013.00 | 2012.86 | 2044.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-17 13:15:00 | 1961.50 | 2008.48 | 2037.86 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1709.97 | 1579.06 | 1712.16 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-02-03 11:15:00 | 1735.00 | 1583.88 | 1711.31 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 14:15:00 | 2199.75 | 1768.57 | 1767.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 10:15:00 | 2238.00 | 1922.28 | 1861.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 09:15:00 | 2708.67 | 2709.06 | 2490.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-28 13:15:00 | 2793.17 | 2710.47 | 2495.40 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 2629.67 | 2743.09 | 2612.74 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-24 10:15:00 | 2687.67 | 2732.60 | 2616.71 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 3305.00 | 3437.78 | 3302.32 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-26 12:15:00 | 3288.00 | 3433.89 | 3302.38 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 12:15:00 | 3103.80 | 3283.85 | 3284.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 3028.00 | 3276.20 | 3280.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 11:15:00 | 2206.60 | 2200.20 | 2441.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-09 09:15:00 | 2144.40 | 2199.13 | 2435.39 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-18 12:15:00 | 2386.80 | 2170.85 | 2365.50 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-17 13:15:00 | 1961.50 | 2024-12-23 09:15:00 | 1732.41 | TARGET | 229.09 |
| BUY | 2025-06-24 10:15:00 | 2687.67 | 2025-06-26 09:15:00 | 2900.54 | TARGET | 212.87 |
| BUY | 2025-05-28 13:15:00 | 2793.17 | 2025-08-07 09:15:00 | 3686.49 | TARGET | 893.32 |
| SELL | 2026-02-09 09:15:00 | 2144.40 | 2026-02-18 12:15:00 | 2386.80 | EXIT_EMA400 | -242.40 |
