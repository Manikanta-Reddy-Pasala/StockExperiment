# Lemon Tree Hotels Ltd. (LEMONTREE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 117.59
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** -10.85
- **Avg P&L per closed trade:** -2.17

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 09:15:00 | 122.08 | 143.73 | 143.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 10:15:00 | 120.33 | 143.50 | 143.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 09:15:00 | 135.61 | 133.78 | 137.43 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-02 09:15:00 | 133.17 | 133.80 | 137.19 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-11 09:15:00 | 124.20 | 120.08 | 124.10 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 15:15:00 | 135.17 | 125.40 | 125.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 09:15:00 | 135.70 | 125.50 | 125.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 11:15:00 | 145.61 | 146.21 | 139.35 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-08 15:15:00 | 147.95 | 146.24 | 139.50 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-10 09:15:00 | 139.16 | 146.08 | 139.68 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 12:15:00 | 127.96 | 138.38 | 138.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 09:15:00 | 125.74 | 137.98 | 138.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 131.20 | 130.50 | 133.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-28 11:15:00 | 128.40 | 132.70 | 133.82 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 133.00 | 132.54 | 133.70 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-01 12:15:00 | 134.51 | 132.55 | 133.70 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 14:15:00 | 139.70 | 134.66 | 134.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 141.51 | 134.78 | 134.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 10:15:00 | 137.66 | 138.58 | 136.83 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-28 15:15:00 | 141.58 | 138.68 | 136.99 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-30 12:15:00 | 136.81 | 138.71 | 137.10 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 15:15:00 | 150.98 | 162.45 | 162.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 11:15:00 | 150.07 | 160.97 | 161.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 161.30 | 158.87 | 160.45 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-17 13:15:00 | 158.75 | 160.81 | 161.10 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 160.72 | 160.58 | 160.96 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-22 10:15:00 | 162.00 | 160.60 | 160.96 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-02 09:15:00 | 133.17 | 2024-09-30 09:15:00 | 121.10 | TARGET | 12.07 |
| BUY | 2025-01-08 15:15:00 | 147.95 | 2025-01-10 09:15:00 | 139.16 | EXIT_EMA400 | -8.79 |
| SELL | 2025-03-28 11:15:00 | 128.40 | 2025-04-01 12:15:00 | 134.51 | EXIT_EMA400 | -6.11 |
| BUY | 2025-04-28 15:15:00 | 141.58 | 2025-04-30 12:15:00 | 136.81 | EXIT_EMA400 | -4.77 |
| SELL | 2025-12-17 13:15:00 | 158.75 | 2025-12-22 10:15:00 | 162.00 | EXIT_EMA400 | -3.25 |
