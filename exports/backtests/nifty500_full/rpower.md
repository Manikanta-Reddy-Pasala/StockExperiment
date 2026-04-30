# Reliance Power Ltd. (RPOWER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 28.61
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** 6.17
- **Avg P&L per closed trade:** 1.23

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-07 13:15:00 | 22.80 | 25.48 | 25.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-07 14:15:00 | 22.60 | 25.45 | 25.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 09:15:00 | 25.05 | 24.05 | 24.65 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 13:15:00 | 28.85 | 25.15 | 25.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 15:15:00 | 29.00 | 25.23 | 25.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 09:15:00 | 27.20 | 27.44 | 26.44 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-18 09:15:00 | 28.70 | 27.40 | 26.52 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 09:15:00 | 27.00 | 27.60 | 26.88 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-05-03 11:15:00 | 26.75 | 27.55 | 26.89 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 09:15:00 | 24.90 | 26.45 | 26.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 09:15:00 | 24.55 | 26.17 | 26.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-10 09:15:00 | 26.35 | 25.66 | 26.00 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 09:15:00 | 32.33 | 26.31 | 26.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-01 09:15:00 | 34.40 | 28.96 | 28.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-07 09:15:00 | 29.82 | 29.86 | 28.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-07 12:15:00 | 31.07 | 29.89 | 28.88 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-08-29 10:15:00 | 30.27 | 31.72 | 30.41 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 09:15:00 | 35.80 | 41.07 | 41.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 12:15:00 | 34.93 | 38.90 | 39.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 11:15:00 | 36.60 | 36.13 | 37.72 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 09:15:00 | 40.01 | 38.60 | 38.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 41.91 | 38.81 | 38.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 12:15:00 | 40.60 | 40.70 | 39.87 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 42.72 | 40.23 | 39.79 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 60.10 | 63.57 | 59.19 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-23 11:15:00 | 62.90 | 63.56 | 59.21 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-25 09:15:00 | 56.78 | 63.24 | 59.30 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 14:15:00 | 45.08 | 56.64 | 56.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 09:15:00 | 43.77 | 56.40 | 56.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 14:15:00 | 47.93 | 47.85 | 50.30 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-18 12:15:00 | 47.45 | 47.90 | 50.19 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-10 09:15:00 | 48.58 | 46.50 | 48.46 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-04-18 09:15:00 | 28.70 | 2024-05-03 11:15:00 | 26.75 | EXIT_EMA400 | -1.95 |
| BUY | 2024-08-07 12:15:00 | 31.07 | 2024-08-22 09:15:00 | 37.64 | TARGET | 6.57 |
| BUY | 2025-05-12 09:15:00 | 42.72 | 2025-05-23 12:15:00 | 51.52 | TARGET | 8.80 |
| BUY | 2025-07-23 11:15:00 | 62.90 | 2025-07-25 09:15:00 | 56.78 | EXIT_EMA400 | -6.12 |
| SELL | 2025-09-18 12:15:00 | 47.45 | 2025-10-10 09:15:00 | 48.58 | EXIT_EMA400 | -1.13 |
