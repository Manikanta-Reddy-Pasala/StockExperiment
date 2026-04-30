# Zee Entertainment Enterprises Ltd. (ZEEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 89.98
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 6 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 4 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / EMA400 exits:** 0 / 7
- **Total realized P&L (per unit):** -23.07
- **Avg P&L per closed trade:** -3.30

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 10:15:00 | 136.48 | 145.52 | 145.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 14:15:00 | 134.16 | 145.17 | 145.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 14:15:00 | 150.11 | 140.13 | 142.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-06 09:15:00 | 134.31 | 140.22 | 141.82 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 11:15:00 | 140.36 | 139.43 | 141.29 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-11 13:15:00 | 136.59 | 139.31 | 141.15 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 137.25 | 135.15 | 138.17 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-26 14:15:00 | 136.55 | 135.22 | 138.16 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 136.99 | 135.26 | 138.13 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-27 12:15:00 | 135.78 | 135.27 | 138.11 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 137.40 | 135.29 | 138.01 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-10-01 09:15:00 | 140.56 | 135.38 | 138.02 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 11:15:00 | 138.90 | 129.58 | 129.56 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 11:15:00 | 125.02 | 129.85 | 129.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 14:15:00 | 124.94 | 129.71 | 129.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 09:15:00 | 128.25 | 128.03 | 128.88 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 10:15:00 | 123.66 | 127.89 | 128.77 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-07 14:15:00 | 129.61 | 127.55 | 128.55 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 15:15:00 | 115.40 | 107.98 | 107.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 118.59 | 108.09 | 108.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 127.30 | 127.93 | 121.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 11:15:00 | 130.31 | 127.98 | 122.12 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 133.80 | 139.58 | 133.38 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-22 15:15:00 | 133.00 | 139.52 | 133.37 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 113.20 | 129.38 | 129.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 14:15:00 | 112.41 | 128.89 | 129.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 10:15:00 | 123.43 | 122.25 | 125.08 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-25 09:15:00 | 120.55 | 122.25 | 125.00 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 119.70 | 117.94 | 120.84 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-22 11:15:00 | 121.35 | 117.97 | 120.84 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-06 09:15:00 | 134.31 | 2024-10-01 09:15:00 | 140.56 | EXIT_EMA400 | -6.25 |
| SELL | 2024-09-11 13:15:00 | 136.59 | 2024-10-01 09:15:00 | 140.56 | EXIT_EMA400 | -3.97 |
| SELL | 2024-09-26 14:15:00 | 136.55 | 2024-10-01 09:15:00 | 140.56 | EXIT_EMA400 | -4.01 |
| SELL | 2024-09-27 12:15:00 | 135.78 | 2024-10-01 09:15:00 | 140.56 | EXIT_EMA400 | -4.78 |
| SELL | 2025-01-06 10:15:00 | 123.66 | 2025-01-07 14:15:00 | 129.61 | EXIT_EMA400 | -5.95 |
| BUY | 2025-06-20 11:15:00 | 130.31 | 2025-07-22 15:15:00 | 133.00 | EXIT_EMA400 | 2.69 |
| SELL | 2025-08-25 09:15:00 | 120.55 | 2025-09-22 11:15:00 | 121.35 | EXIT_EMA400 | -0.80 |
