# Motherson Sumi Wiring India Ltd. (MSUMI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 40.54
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 3 |
| ENTRY1 | 7 |
| ENTRY2 | 2 |
| EXIT | 7 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 4 / 5
- **Target hits / EMA400 exits:** 4 / 5
- **Total realized P&L (per unit):** 6.98
- **Avg P&L per closed trade:** 0.78

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 11:15:00 | 39.23 | 41.12 | 41.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 12:15:00 | 39.00 | 41.10 | 41.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-20 14:15:00 | 40.37 | 40.32 | 40.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-22 11:15:00 | 39.73 | 40.31 | 40.62 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-11-23 10:15:00 | 40.77 | 40.28 | 40.60 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-01 10:15:00 | 41.40 | 40.71 | 40.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-05 09:15:00 | 43.03 | 40.86 | 40.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 10:15:00 | 41.60 | 41.66 | 41.27 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-30 09:15:00 | 42.47 | 41.45 | 41.24 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-11 09:15:00 | 44.23 | 46.05 | 44.69 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-05 13:15:00 | 42.70 | 45.16 | 45.16 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-11 14:15:00 | 45.86 | 45.16 | 45.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-12 13:15:00 | 46.05 | 45.20 | 45.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 09:15:00 | 48.13 | 48.30 | 47.20 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-12 09:15:00 | 49.40 | 48.31 | 47.35 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-07-23 12:15:00 | 47.71 | 48.63 | 47.73 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 11:15:00 | 46.71 | 47.75 | 47.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 13:15:00 | 46.67 | 47.73 | 47.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-24 13:15:00 | 48.19 | 46.94 | 47.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-01 14:15:00 | 46.67 | 47.14 | 47.30 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-10 14:15:00 | 43.59 | 42.37 | 43.29 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 38.00 | 35.89 | 35.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 14:15:00 | 38.10 | 36.18 | 36.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 13:15:00 | 39.43 | 39.45 | 38.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 09:15:00 | 39.71 | 39.45 | 38.39 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 40.49 | 41.54 | 40.37 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-25 15:15:00 | 40.70 | 41.52 | 40.37 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-28 09:15:00 | 40.19 | 41.51 | 40.37 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 15:15:00 | 37.65 | 39.65 | 39.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 37.21 | 39.63 | 39.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 10:15:00 | 39.51 | 39.50 | 39.58 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 14:15:00 | 41.88 | 39.66 | 39.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 15:15:00 | 42.00 | 39.68 | 39.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 14:15:00 | 45.73 | 45.82 | 43.90 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-01 13:15:00 | 45.90 | 45.81 | 43.95 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 44.54 | 45.67 | 44.26 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-13 11:15:00 | 44.82 | 45.66 | 44.26 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-10-14 11:15:00 | 44.04 | 45.59 | 44.28 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 43.32 | 46.32 | 46.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 42.95 | 46.29 | 46.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 11:15:00 | 45.14 | 45.06 | 45.61 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-05 09:15:00 | 44.50 | 45.07 | 45.60 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 40.02 | 39.23 | 40.66 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-22 11:15:00 | 40.67 | 39.27 | 40.66 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-22 11:15:00 | 39.73 | 2023-11-23 10:15:00 | 40.77 | EXIT_EMA400 | -1.03 |
| BUY | 2024-01-30 09:15:00 | 42.47 | 2024-02-01 09:15:00 | 46.16 | TARGET | 3.69 |
| BUY | 2024-07-12 09:15:00 | 49.40 | 2024-07-23 12:15:00 | 47.71 | EXIT_EMA400 | -1.69 |
| SELL | 2024-10-01 14:15:00 | 46.67 | 2024-10-04 09:15:00 | 44.77 | TARGET | 1.89 |
| BUY | 2025-06-20 09:15:00 | 39.71 | 2025-07-09 09:15:00 | 43.69 | TARGET | 3.98 |
| BUY | 2025-07-25 15:15:00 | 40.70 | 2025-07-28 09:15:00 | 40.19 | EXIT_EMA400 | -0.51 |
| BUY | 2025-10-01 13:15:00 | 45.90 | 2025-10-14 11:15:00 | 44.04 | EXIT_EMA400 | -1.86 |
| BUY | 2025-10-13 11:15:00 | 44.82 | 2025-10-14 11:15:00 | 44.04 | EXIT_EMA400 | -0.78 |
| SELL | 2026-02-05 09:15:00 | 44.50 | 2026-03-02 09:15:00 | 41.21 | TARGET | 3.29 |
