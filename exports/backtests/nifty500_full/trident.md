# Trident Ltd. (TRIDENT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 25.97
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -7.02
- **Avg P&L per closed trade:** -1.76

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 14:15:00 | 35.10 | 36.08 | 36.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 15:15:00 | 34.80 | 36.07 | 36.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-07 09:15:00 | 36.60 | 35.82 | 35.94 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-11-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 12:15:00 | 38.35 | 36.06 | 36.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-05 12:15:00 | 40.40 | 36.87 | 36.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-09 09:15:00 | 43.95 | 44.16 | 41.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-16 10:15:00 | 45.80 | 43.86 | 42.03 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-02-28 10:15:00 | 42.35 | 43.99 | 42.56 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-03-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 09:15:00 | 37.90 | 41.72 | 41.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-15 10:15:00 | 37.30 | 41.68 | 41.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-02 12:15:00 | 39.75 | 39.62 | 40.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-15 09:15:00 | 38.85 | 39.96 | 40.47 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 12:15:00 | 40.15 | 39.66 | 40.22 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-04-25 12:15:00 | 40.30 | 39.71 | 40.21 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 15:15:00 | 29.53 | 27.82 | 27.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 14:15:00 | 33.50 | 27.95 | 27.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 30.26 | 30.45 | 29.57 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-17 09:15:00 | 31.00 | 30.45 | 29.60 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 29.66 | 30.43 | 29.66 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-19 12:15:00 | 29.32 | 30.42 | 29.65 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 09:15:00 | 27.96 | 30.29 | 30.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 13:15:00 | 27.85 | 30.20 | 30.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 29.48 | 28.67 | 29.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-26 10:15:00 | 28.59 | 29.21 | 29.33 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-23 09:15:00 | 29.03 | 28.50 | 28.84 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-02-16 10:15:00 | 45.80 | 2024-02-28 10:15:00 | 42.35 | EXIT_EMA400 | -3.45 |
| SELL | 2024-04-15 09:15:00 | 38.85 | 2024-04-25 12:15:00 | 40.30 | EXIT_EMA400 | -1.45 |
| BUY | 2025-06-17 09:15:00 | 31.00 | 2025-06-19 12:15:00 | 29.32 | EXIT_EMA400 | -1.68 |
| SELL | 2025-09-26 10:15:00 | 28.59 | 2025-10-23 09:15:00 | 29.03 | EXIT_EMA400 | -0.44 |
