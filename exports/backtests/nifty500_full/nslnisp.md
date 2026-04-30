# NMDC Steel Ltd. (NSLNISP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 42.72
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT3 | 5 |
| ENTRY1 | 6 |
| ENTRY2 | 3 |
| EXIT | 6 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 3 / 6
- **Target hits / EMA400 exits:** 2 / 7
- **Total realized P&L (per unit):** -13.42
- **Avg P&L per closed trade:** -1.49

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 14:15:00 | 41.05 | 50.19 | 50.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-26 09:15:00 | 40.10 | 50.00 | 50.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 14:15:00 | 44.05 | 43.63 | 45.64 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 09:15:00 | 52.10 | 46.38 | 46.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-23 09:15:00 | 53.45 | 49.12 | 48.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 14:15:00 | 58.80 | 59.52 | 54.90 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-15 09:15:00 | 66.00 | 59.51 | 55.25 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 11:15:00 | 58.35 | 60.82 | 58.15 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-03-12 13:15:00 | 58.80 | 60.78 | 58.15 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-03-13 09:15:00 | 55.65 | 60.69 | 58.15 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 10:15:00 | 59.70 | 61.02 | 61.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 13:15:00 | 59.58 | 60.98 | 61.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 09:15:00 | 60.24 | 58.73 | 59.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-07-19 09:15:00 | 56.66 | 58.95 | 59.57 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 58.80 | 58.26 | 59.07 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-07-29 13:15:00 | 58.01 | 58.27 | 59.06 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-08-01 09:15:00 | 59.00 | 58.24 | 58.98 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 09:15:00 | 39.00 | 37.09 | 37.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 11:15:00 | 39.96 | 37.29 | 37.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 38.20 | 38.56 | 37.98 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-25 09:15:00 | 39.26 | 38.35 | 37.99 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-25 09:15:00 | 38.85 | 39.42 | 38.97 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 14:15:00 | 36.96 | 38.61 | 38.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 36.66 | 38.57 | 38.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 41.80 | 37.76 | 38.14 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-19 12:15:00 | 39.77 | 38.42 | 38.45 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 39.77 | 38.42 | 38.45 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-08-19 13:15:00 | 39.61 | 38.44 | 38.46 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 10:15:00 | 40.46 | 38.50 | 38.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 12:15:00 | 40.60 | 38.54 | 38.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 38.44 | 38.68 | 38.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-02 10:15:00 | 39.26 | 38.53 | 38.52 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 39.26 | 38.53 | 38.52 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-02 11:15:00 | 39.54 | 38.54 | 38.53 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 43.02 | 44.30 | 42.91 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-10-17 14:15:00 | 42.82 | 44.28 | 42.91 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 14:15:00 | 41.99 | 43.05 | 43.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 41.80 | 42.99 | 43.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 41.53 | 41.42 | 42.05 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 45.24 | 42.44 | 42.44 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 40.37 | 42.49 | 42.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 39.86 | 42.47 | 42.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 10:15:00 | 42.54 | 41.88 | 42.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-02 09:15:00 | 40.70 | 42.03 | 42.22 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-02 14:15:00 | 42.30 | 42.00 | 42.20 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 14:15:00 | 40.93 | 39.63 | 39.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 41.88 | 39.67 | 39.65 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-02-15 09:15:00 | 66.00 | 2024-03-13 09:15:00 | 55.65 | EXIT_EMA400 | -10.35 |
| BUY | 2024-03-12 13:15:00 | 58.80 | 2024-03-13 09:15:00 | 55.65 | EXIT_EMA400 | -3.15 |
| SELL | 2024-07-19 09:15:00 | 56.66 | 2024-08-01 09:15:00 | 59.00 | EXIT_EMA400 | -2.34 |
| SELL | 2024-07-29 13:15:00 | 58.01 | 2024-08-01 09:15:00 | 59.00 | EXIT_EMA400 | -0.99 |
| BUY | 2025-06-25 09:15:00 | 39.26 | 2025-07-25 09:15:00 | 38.85 | EXIT_EMA400 | -0.41 |
| SELL | 2025-08-19 12:15:00 | 39.77 | 2025-08-19 13:15:00 | 39.61 | EXIT_EMA400 | 0.16 |
| BUY | 2025-09-02 10:15:00 | 39.26 | 2025-09-03 11:15:00 | 41.48 | TARGET | 2.22 |
| BUY | 2025-09-02 11:15:00 | 39.54 | 2025-09-03 11:15:00 | 42.58 | TARGET | 3.04 |
| SELL | 2026-02-02 09:15:00 | 40.70 | 2026-02-02 14:15:00 | 42.30 | EXIT_EMA400 | -1.60 |
