# Zee Entertainment Enterprises Ltd. (ZEEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 89.74
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -10.59
- **Avg P&L per closed trade:** -2.65

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-19 14:15:00 | 234.00 | 262.42 | 262.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 09:15:00 | 208.30 | 261.62 | 262.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-29 12:15:00 | 150.30 | 149.50 | 164.40 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-30 14:15:00 | 146.70 | 149.50 | 163.75 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-05-29 09:15:00 | 153.30 | 144.01 | 152.98 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 15:15:00 | 137.65 | 129.94 | 129.93 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 12:15:00 | 124.99 | 130.14 | 130.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 14:15:00 | 124.80 | 130.04 | 130.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 09:15:00 | 128.26 | 128.03 | 128.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 10:15:00 | 123.68 | 127.88 | 128.88 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-07 14:15:00 | 129.66 | 127.55 | 128.65 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 118.57 | 108.10 | 108.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 10:15:00 | 121.40 | 108.91 | 108.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 127.30 | 127.95 | 121.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 11:15:00 | 130.31 | 127.99 | 122.15 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 133.84 | 139.58 | 133.39 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-22 15:15:00 | 133.00 | 139.52 | 133.39 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 113.20 | 129.38 | 129.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 14:15:00 | 112.44 | 128.88 | 129.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 10:15:00 | 123.47 | 122.25 | 125.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-25 09:15:00 | 120.60 | 122.26 | 125.01 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 119.70 | 117.93 | 120.84 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-22 11:15:00 | 121.30 | 117.97 | 120.84 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-04-30 14:15:00 | 146.70 | 2024-05-29 09:15:00 | 153.30 | EXIT_EMA400 | -6.60 |
| SELL | 2025-01-06 10:15:00 | 123.68 | 2025-01-07 14:15:00 | 129.66 | EXIT_EMA400 | -5.98 |
| BUY | 2025-06-20 11:15:00 | 130.31 | 2025-07-22 15:15:00 | 133.00 | EXIT_EMA400 | 2.69 |
| SELL | 2025-08-25 09:15:00 | 120.60 | 2025-09-22 11:15:00 | 121.30 | EXIT_EMA400 | -0.70 |
